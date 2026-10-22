"""
Generate the credentials for a real (non-simulated) deployment:

  TLS          a private CA and one server certificate for the SuperLink (SANs: localhost, 127.0.0.1, ::1,
               plus any --host given). Every SuperNode verifies the SuperLink against ca.crt.
  hospitals    one P-384 EC key pair per hospital SuperNode in OpenSSH format (what `flwr supernode register`
               and `flower-supernode --auth-supernode-private-key` read). The SuperLink only accepts nodes whose
               public key was registered, so a hospital is identified by its key.

Everything lands in deploy/secrets/ (gitignored). Private keys are written 0600 and each hospital should
receive only its own key. These are prototype credentials for a private network: use your organisation's
PKI and key management in production. Existing files are never overwritten without --force.

    python models/objectiveC/deploy/credentials.py [--out deploy/secrets] [--host sl.example.org] [--force]
"""
import argparse
import datetime as dt
import ipaddress
import os
import sys
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "secrets"
VALID_DAYS = 365


def _write(path: Path, data: bytes, private: bool, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to replace it")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _pem_private(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _ssh_private(key) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
                             serialization.NoEncryption())


def _ssh_public(key) -> bytes:
    return key.public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH)


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "carepath-fl"),
                      x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def make_tls(out: Path, hosts: list[str], force: bool = False) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = (x509.CertificateBuilder().subject_name(_name("carepath-fl CA")).issuer_name(_name("carepath-fl CA"))
               .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
               .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=VALID_DAYS))
               .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
               .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                                            key_encipherment=False, data_encipherment=False, key_agreement=False,
                                            encipher_only=False, decipher_only=False), critical=True)
               .sign(ca_key, hashes.SHA256()))

    sans = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            x509.IPAddress(ipaddress.ip_address("::1"))]
    for h in hosts:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            sans.append(x509.DNSName(h))
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(_name("carepath-fl SuperLink")).issuer_name(ca_cert.subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=VALID_DAYS))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH, ExtendedKeyUsageOID.CLIENT_AUTH]),
                           critical=False)
            .sign(ca_key, hashes.SHA256()))

    _write(out / "ca.crt", ca_cert.public_bytes(serialization.Encoding.PEM), False, force)
    _write(out / "ca.key", _pem_private(ca_key), True, force)
    _write(out / "server.pem", cert.public_bytes(serialization.Encoding.PEM), False, force)
    _write(out / "server.key", _pem_private(key), True, force)


def make_hospital_keys(out: Path, hospitals: list[str], force: bool = False) -> None:
    keys = out / "hospitals"
    keys.mkdir(exist_ok=True)
    for h in hospitals:
        private = ec.generate_private_key(ec.SECP384R1())
        _write(keys / f"{h}", _ssh_private(private), True, force)
        _write(keys / f"{h}.pub", _ssh_public(private.public_key()), False, force)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--host", action="append", default=[], help="extra SuperLink DNS name or IP for the certificate")
    ap.add_argument("--hospitals", default=None, help="comma-separated ids (default: those in hospital_master.csv)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.hospitals:
        hospitals = args.hospitals.split(",")
    else:
        sys.path.insert(0, str(HERE.parent))
        from carepath_fl.common.partition import hospital_ids
        hospitals = hospital_ids()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    make_tls(out, args.host, args.force)
    make_hospital_keys(out, hospitals, args.force)
    print(f"wrote TLS files and {len(hospitals)} hospital key pairs to {out}")


if __name__ == "__main__":
    main()
