import sys, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _bootstrap import use_branch_library  # noqa
use_branch_library()
import torch
from knowlytix.core.config import GeometryConfig
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.store import GMSExpertStore
from knowlytix.harness.testing.hallucination import HallucinationOracle

sp = "/home/user/jupyterlab/forgeloop/beyond-chunk-and-pray/code/data/gms_annual_report_store"
g = json.load(open(sp + "/model_dims.json"))["geometry"]
store = GMSExpertStore(DocGMSConfig(store_path=sp, geometry=GeometryConfig(
    d_v=g["d_v"], d_u=g["d_u"], m=g["m"], d=g["d"])), device=torch.device("cuda"))
store.load()
o = HallucinationOracle(store=store)
tests = [("cloud platform", "has_revenue", "120.0"),
         ("cloud platform", "has_revenue", "120"),
         ("cloud platform", "has_revenue", "999.0"),
         ("revenue", "has_fy2024", "320.0"),
         ("total assets", "has_amount", "540.0")]
for h, r, t in tests:
    v = o.assess_claim(h, r, t)
    print(f"{h!r:18}{r:13}{t:8} -> passed={v.passed} "
          f"band={getattr(v,'band',None)} code={getattr(v,'failure_code',None)} "
          f"score={getattr(v,'score',None)}")
