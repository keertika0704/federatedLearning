"""
Run the federation the way a real deployment would, on one machine: a SuperLink with TLS and mandatory node
authentication, and one SuperNode per hospital, each connecting over TLS with its own private key. Unlike
`flwr run` in simulation, the hospitals are separate processes and the SuperLink only talks to nodes whose
public key was registered.

    python models/objectiveC/deploy/local_federation.py run --task disease
    python models/objectiveC/deploy/local_federation.py check-security

`run` starts everything, runs the task, stops everything. `check-security` starts the SuperLink and verifies
that it refuses a node with an unregistered key, a node that skips TLS, and a node that does not trust the CA.

All Flower state goes to deploy/work/ (via FLWR_HOME), never to ~/.flwr. Credentials come from deploy/secrets/
(create them with credentials.py). Run with the Python environment that has flwr installed; its bin directory
is put on PATH so the flower-* commands are found.
"""
import argparse
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
REPO = APP_DIR.parents[1]
SECRETS = HERE / "secrets"
WORK = HERE / "work"
PROFILE = "carepath-deployment"
# not 8000: that is the web app's port (uvicorn backend.main:app), and both can run at the same time
HOST, FLEET_PORT = "127.0.0.1", 9092
NODE_PORT0 = 9094


def hospital_ids() -> list[str]:
    sys.path.insert(0, str(APP_DIR))
    from carepath_fl.common.partition import hospital_ids as ids
    return ids()


def port_in_use(port: int, host: str = HOST) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_port(port: int, timeout: float = 60.0, host: str = HOST) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"nothing listening on {host}:{port} after {timeout:.0f}s")


class Deployment:
    """Owns the SuperLink and SuperNode processes; use as a context manager so they are always stopped."""

    def __init__(self, work: Path = WORK, secrets: Path = SECRETS):
        self.work, self.secrets = work, secrets
        self.procs: list[subprocess.Popen] = []
        self.registry = work / "registry.json"
        self.env = {**os.environ, "FLWR_HOME": str(work / "flwr_home"), "CAREPATH_ROOT": str(REPO),
                    "CAREPATH_NODE_REGISTRY": str(self.registry),
                    "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}

    # -------------------------------------------------------------- lifecycle
    def __enter__(self):
        for f in ("ca.crt", "server.pem", "server.key"):
            if not (self.secrets / f).is_file():
                raise FileNotFoundError(f"{self.secrets / f} missing: run deploy/credentials.py first")
        if port_in_use(FLEET_PORT):                         # else wait_for_port below would accept someone else's server
            raise RuntimeError(f"port {FLEET_PORT} is already in use (a previous run still alive?); "
                               f"stop it before starting the deployment")
        shutil.rmtree(self.work, ignore_errors=True)       # a fresh state every run: no stale nodes or runs
        (self.work / "logs").mkdir(parents=True)
        home = self.work / "flwr_home"
        home.mkdir()
        (home / "config.toml").write_text(
            f'[superlink]\ndefault = "{PROFILE}"\n\n[superlink.{PROFILE}]\n'
            f'address = "{HOST}:{FLEET_PORT}"\nroot-certificates = "{self.secrets / "ca.crt"}"\n')
        self.spawn("superlink", ["flower-superlink",
                                 "--ssl-ca-certfile", self.secrets / "ca.crt",
                                 "--ssl-certfile", self.secrets / "server.pem",
                                 "--ssl-keyfile", self.secrets / "server.key",
                                 "--host", HOST, "--port", FLEET_PORT,
                                 "--enable-supernode-auth", "--disable-runtime-dependency-installation",
                                 "--database", self.work / "superlink.db"])
        wait_for_port(FLEET_PORT)
        return self

    def __exit__(self, *exc):
        for p in reversed(self.procs):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in self.procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)

    def spawn(self, name: str, cmd: list, env: dict | None = None) -> subprocess.Popen:
        log = open(self.work / "logs" / f"{name}.log", "wb")
        p = subprocess.Popen([str(c) for c in cmd], env=env or self.env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        self.procs.append(p)
        return p

    def flwr(self, *args, check: bool = True, timeout: float = 120) -> subprocess.CompletedProcess:
        r = subprocess.run(["flwr", *[str(a) for a in args]], env=self.env, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode:
            raise RuntimeError(f"flwr {' '.join(map(str, args))} failed:\n{r.stdout}\n{r.stderr}")
        return r

    # -------------------------------------------------------------- hospitals
    def register_hospitals(self, hospitals: list[str]) -> dict[int, str]:
        """Register each hospital's public key with the SuperLink; remember which node id is which hospital."""
        registry = {}
        for h in hospitals:
            r = self.flwr("supernode", "register", self.secrets / "hospitals" / f"{h}.pub", PROFILE,
                          "--name", h, "--format", "json")
            m = re.search(r'"node[-_]id"\s*:\s*"?(\d+)"?', r.stdout)
            if not m:
                raise RuntimeError(f"could not read a node id from: {r.stdout!r}")
            registry[int(m.group(1))] = h
        self.registry.write_text(json.dumps({str(k): v for k, v in registry.items()}, indent=2))
        return registry

    def start_node(self, index: int, hospital: str, key: Path | None, *, name: str | None = None,
                   insecure: bool = False, ca: Path | None = None, retries: int | None = None) -> subprocess.Popen:
        node_home = self.work / f"node_{name or hospital}"
        node_home.mkdir(exist_ok=True)
        cmd = ["flower-supernode", "--superlink", f"{HOST}:{FLEET_PORT}", "--port", NODE_PORT0 + index,
               "--node-config", f"partition-id={index} num-partitions=10"]
        cmd += ["--insecure"] if insecure else ["--root-certificates", ca or self.secrets / "ca.crt"]
        if key:
            cmd += ["--auth-supernode-private-key", key]
        if retries is not None:
            cmd += ["--max-retries", retries, "--max-wait-time", 30]
        return self.spawn(f"node_{name or hospital}", cmd, {**self.env, "FLWR_HOME": str(node_home)})

    def start_hospitals(self, hospitals: list[str]) -> None:
        for i, h in enumerate(hospitals):
            self.start_node(i, h, self.secrets / "hospitals" / h)

    def wait_online(self, n: int, timeout: float = 120) -> None:
        end = time.time() + timeout
        while time.time() < end:
            r = self.flwr("supernode", "list", PROFILE, "--format", "json", check=False)
            if len(re.findall(r'"online"', r.stdout)) >= n:
                return
            time.sleep(2)
        raise TimeoutError(f"fewer than {n} SuperNodes online after {timeout:.0f}s:\n{r.stdout}")

    def online_nodes(self) -> int:
        r = self.flwr("supernode", "list", PROFILE, "--format", "json", check=False)
        return len(re.findall(r'"online"', r.stdout))

    def run_app(self, task: str, extra: str = "") -> subprocess.CompletedProcess:
        return self.flwr("run", APP_DIR, PROFILE, "--run-config", f'task="{task}" {extra}'.strip(), "--stream",
                         check=False, timeout=3600)


# ------------------------------------------------------------------ commands
def cmd_run(args) -> int:
    hospitals = hospital_ids()
    with Deployment() as dep:
        reg = dep.register_hospitals(hospitals)
        print(f"registered {len(reg)} hospital SuperNodes: " + ", ".join(f"{h}={n}" for n, h in reg.items()))
        dep.start_hospitals(hospitals)
        dep.wait_online(len(hospitals))
        print(f"{len(hospitals)} SuperNodes online over TLS; running task {args.task!r} ...")
        r = dep.run_app(args.task, args.run_config)
        print(re.sub(r"\x1b\[[0-9;]*m", "", r.stdout)[-4000:], r.stderr[-1500:] if r.returncode else "")
        print(f"flwr run exited with {r.returncode}; logs in {WORK / 'logs'}")
        return r.returncode


def node_log(dep: "Deployment", name: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", (dep.work / "logs" / f"node_{name}.log").read_text(errors="replace"))


def was_refused(log: str, *needles: str) -> bool:
    """The node started fine (its own HTTP API came up) and then could not connect, for the stated reason."""
    return "failed to start" not in log and any(n in log.lower() for n in needles)


def cmd_check_security(args) -> int:
    """Each refusal is checked three ways: the node really started (own port, no startup crash), its log shows
    the connection was refused for the expected reason, and the number of online nodes did not grow. The
    unregistered-key case also has a control: registering that same key afterwards lets the node connect."""
    hospitals = hospital_ids()
    results = []
    with Deployment() as dep:
        dep.register_hospitals(hospitals[:1])                    # only H001 is registered
        dep.start_node(0, "H001", dep.secrets / "hospitals" / "H001")
        dep.wait_online(1)
        results.append(("registered hospital connects over TLS", dep.online_nodes() == 1))

        sys.path.insert(0, str(HERE))
        from credentials import make_hospital_keys, make_tls

        # 1. a key pair nobody registered (own port 9095)
        outsider = dep.work / "outsider"
        outsider.mkdir()
        make_hospital_keys(outsider, ["H002"])
        key, pub = outsider / "hospitals" / "H002", outsider / "hospitals" / "H002.pub"
        dep.start_node(1, "H002", key, name="unregistered", retries=2)
        time.sleep(25)
        results.append(("unregistered key is refused", dep.online_nodes() == 1 and
                        was_refused(node_log(dep, "unregistered"), "connection attempt failed", "unauthenticated",
                                    "permission", "not registered")))
        # control: same key, now registered -> the node connects, so registration was the only difference
        dep.flwr("supernode", "register", pub, PROFILE, "--name", "H002")
        dep.start_node(2, "H002", key, name="registered_later")
        try:
            dep.wait_online(2, timeout=60)
            results.append(("control: the same key connects once registered", True))
        except TimeoutError:
            results.append(("control: the same key connects once registered", False))
        online_before = dep.online_nodes()

        # 2. no TLS. (a) the SuperLink itself: a plaintext HTTP request to its TLS-only Fleet API gets no answer
        try:
            conn = http.client.HTTPConnection(HOST, FLEET_PORT, timeout=5)
            conn.request("GET", "/")
            plaintext_served = conn.getresponse().status < 400
        except (OSError, http.client.HTTPException):
            plaintext_served = False
        results.append(("SuperLink does not answer plaintext HTTP on its Fleet port", not plaintext_served))
        # (b) the SuperNode refuses to present its private key over an insecure channel
        p = dep.start_node(3, "H001", dep.secrets / "hospitals" / "H001", name="plaintext", insecure=True, retries=2)
        time.sleep(15)
        results.append(("SuperNode refuses to authenticate over plaintext", dep.online_nodes() == online_before
                        and p.poll() not in (None, 0) and "requires a secure tls connection" in node_log(dep, "plaintext").lower()))

        # 3. TLS, but trusting a different CA than the SuperLink's (own port)
        other = dep.work / "othertls"
        other.mkdir()
        make_tls(other, [])
        dep.start_node(4, "H001", dep.secrets / "hospitals" / "H001", name="wrongca", ca=other / "ca.crt", retries=2)
        time.sleep(20)
        results.append(("node that does not trust the SuperLink's CA does not connect",
                        dep.online_nodes() == online_before and
                        was_refused(node_log(dep, "wrongca"), "connection attempt failed", "certificate", "ssl")))
    ok = all(v for _, v in results)
    for name, v in results:
        print(("PASS  " if v else "FAIL  ") + name)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start the deployment, run a task, stop")
    r.add_argument("--task", default="disease", choices=["disease", "treatment", "atypical", "outbreak"])
    r.add_argument("--run-config", default="", help='extra run-config, e.g. "num-server-rounds=10"')
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("check-security", help="verify the SuperLink refuses unregistered / plaintext / untrusted nodes")
    c.set_defaults(fn=cmd_check_security)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
