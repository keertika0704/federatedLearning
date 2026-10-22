"""
Which hospital is behind which authenticated SuperNode.

The SuperLink (with --enable-supernode-auth) only accepts nodes whose public key was registered, and gives
each a node id. deploy/local_federation.py writes a registry {node id: hospital id} at registration time,
and the ServerApp reads it from $CAREPATH_NODE_REGISTRY. With a registry the server

  * refuses to run when a connected node is not in it, or when a registered hospital is missing from a
    stage that needs every hospital (the pooled statistics would silently drop one),
  * takes each reply's sender from the authenticated message metadata, never from the payload,
  * rejects a reply whose payload claims to be a different hospital than its sender (a hospital cannot
    submit summaries as another hospital), and a hospital that answers twice.

Without a registry (plain simulation) nothing is enforced and a notice is logged.
"""
import json
import os
from pathlib import Path

ENV = "CAREPATH_NODE_REGISTRY"


class IdentityError(RuntimeError):
    """A reply or node does not match the registered hospital identities."""


class NodeRegistry:
    def __init__(self, hospital_by_node: dict[int, str]):
        if len(set(hospital_by_node.values())) != len(hospital_by_node):
            raise ValueError("a hospital is registered under more than one node id")
        self.hospital_by_node = dict(hospital_by_node)

    @classmethod
    def from_env(cls) -> "NodeRegistry | None":
        path = os.environ.get(ENV)
        if not path:
            return None
        p = Path(path)
        if not p.is_file():
            raise IdentityError(f"{ENV}={path} does not exist; register the hospitals first")
        return cls({int(k): v for k, v in json.loads(p.read_text()).items()})

    @property
    def node_ids(self) -> set[int]:
        return set(self.hospital_by_node)

    def check_connected(self, node_ids) -> None:
        unknown = set(node_ids) - self.node_ids
        if unknown:
            raise IdentityError(f"connected nodes {sorted(unknown)} are not registered hospitals")

    def check_complete(self, node_ids) -> None:
        self.check_connected(node_ids)
        missing = self.node_ids - set(node_ids)
        if missing:
            raise IdentityError("hospitals not connected: "
                                + ", ".join(sorted(self.hospital_by_node[n] for n in missing)))

    def check_reply(self, sender_node_id: int, claimed_hospital: str | None) -> str:
        """The hospital behind an authenticated sender; raises if the payload claims to be someone else."""
        hospital = self.hospital_by_node.get(sender_node_id)
        if hospital is None:
            raise IdentityError(f"reply from unregistered node {sender_node_id}")
        if claimed_hospital is not None and claimed_hospital != hospital:
            raise IdentityError(f"node {sender_node_id} is {hospital} but its reply claims to be {claimed_hospital}")
        return hospital

    def check_replies(self, senders: list[int], claims: list[str | None]) -> list[str]:
        hospitals = [self.check_reply(s, c) for s, c in zip(senders, claims)]
        if len(set(hospitals)) != len(hospitals):
            raise IdentityError("a hospital answered more than once in the same stage")
        return hospitals
