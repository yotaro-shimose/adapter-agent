from dataclasses import dataclass
from oai_utils import AgentsSDKModel


@dataclass(kw_only=True)
class BaseAgent[M: AgentsSDKModel]:
    model: M
