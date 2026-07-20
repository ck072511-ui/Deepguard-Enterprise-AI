"""
DeepGuard — Final Project Verification Script
Verifies all modules can be imported and core components instantiate correctly.
"""
import sys
import traceback
from pathlib import Path

# Add project root to sys.path so modules resolve correctly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

results = []

def check(name, fn):
    try:
        fn()
        results.append(("PASS", name))
        print(f"  PASS  {name}")
    except Exception as e:
        results.append(("FAIL", name, str(e)))
        print(f"  FAIL  {name}: {e}")

print("=" * 60)
print("DeepGuard — Module Verification")
print("=" * 60)

# Core imports
check("Import: models.config", lambda: __import__("models.config"))
check("Import: models.factory", lambda: __import__("models.factory"))
check("Import: database.models", lambda: __import__("database.models"))
check("Import: database.session", lambda: __import__("database.session"))
check("Import: schemas.responses.detection", lambda: __import__("schemas.responses.detection"))
check("Import: schemas.responses.common", lambda: __import__("schemas.responses.common"))
check("Import: utils.explainability", lambda: __import__("utils.explainability"))
check("Import: core.exceptions.api_exceptions", lambda: __import__("core.exceptions.api_exceptions"))
check("Import: repositories.sqlite.detection", lambda: __import__("repositories.sqlite.detection"))
check("Import: repositories.sqlite.model", lambda: __import__("repositories.sqlite.model"))
check("Import: training.callbacks.base", lambda: __import__("training.callbacks.base"))
check("Import: training.losses.focal", lambda: __import__("training.losses.focal"))
check("Import: training.schedulers.cosine_warmup", lambda: __import__("training.schedulers.cosine_warmup"))
check("Import: datasets.augmentations.train_transforms", lambda: __import__("datasets.augmentations.train_transforms"))
check("Import: datasets.augmentations.val_transforms", lambda: __import__("datasets.augmentations.val_transforms"))

# Model instantiation
def check_model():
    from models.config import FullModelConfig
    from models.factory import ModelFactory
    import yaml
    from pathlib import Path
    cfg_path = Path("configs/model_config.yaml")
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    config = FullModelConfig(**raw)
    model = ModelFactory.create_model(config)
    assert model is not None

check("ModelFactory.create_model()", check_model)

# Explainability engine
def check_xai():
    import numpy as np
    from utils.explainability import ExplainabilityEngine
    # ExplainabilityEngine uses class/static methods — just verify it's importable and has key methods
    assert hasattr(ExplainabilityEngine, 'generate_synthetic_map')
    assert hasattr(ExplainabilityEngine, 'to_base64_jpeg')
    # Verify a static call works with required argument
    dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)
    result = ExplainabilityEngine.generate_synthetic_map(dummy_face, is_fake=True)
    assert result is not None

check("ExplainabilityEngine instantiation", check_xai)

# Schema validation
def check_schemas():
    from schemas.responses.detection import DetectionResponse
    from schemas.responses.common import PaginatedResponse
    from schemas.responses.health import HealthCheckResponse

check("Pydantic schemas importable", check_schemas)

# Config loading
def check_configs():
    import yaml
    from pathlib import Path
    for cfg in ["model_config.yaml", "api_config.yaml", "training_config.yaml"]:
        p = Path("configs") / cfg
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict)

check("YAML configs loadable", check_configs)

# Summary
print()
print("=" * 60)
passed = sum(1 for r in results if r[0] == "PASS")
failed = sum(1 for r in results if r[0] == "FAIL")
print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
print("=" * 60)

if failed > 0:
    print("\nFailed checks:")
    for r in results:
        if r[0] == "FAIL":
            print(f"  - {r[1]}: {r[2]}")
    sys.exit(1)
else:
    print("All module checks passed!")
    sys.exit(0)
