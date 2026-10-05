import torch
from src.augmentation import augment_flair_and_seg
from src.model import MultiTask3DCNN
from src.slot_metrics import majority_classes, majority_predictions, evaluate_slot_predictions


def test_multitask_model_shapes():
    model = MultiTask3DCNN(4, 0.0)
    out = model(torch.randn(2, 1, 32, 32, 32))
    assert out["edema"].shape == (2, 2)
    assert out["hemisphere"].shape == (2, 2)
    assert out["extent"].shape == (2, 3)


def test_augmentation_preserves_segmentation_classes():
    x = torch.randn(2, 1, 24, 24, 24)
    y = torch.randint(0, 4, (2, 1, 24, 24, 24))
    xa, ya = augment_flair_and_seg(x, y, probability=1.0)
    assert xa.shape == x.shape and ya.shape == y.shape
    assert set(ya.unique().tolist()).issubset({0, 1, 2, 3})


def test_majority_baseline_is_explicit_and_train_derived():
    training = [
        {"edema": 1, "enhancing": 1, "necrotic": 1, "hemisphere": 0, "extent": 0},
        {"edema": 1, "enhancing": 1, "necrotic": 1, "hemisphere": 1, "extent": 1},
        {"edema": 1, "enhancing": 0, "necrotic": 1, "hemisphere": 1, "extent": 2},
    ]
    test = [
        {"edema": 1, "enhancing": 0, "necrotic": 1, "hemisphere": 0, "extent": 2},
        {"edema": 1, "enhancing": 1, "necrotic": 1, "hemisphere": 1, "extent": 1},
    ]
    classes = majority_classes(training)
    pred = majority_predictions(classes, len(test))
    result = evaluate_slot_predictions(pred, test)
    assert classes["edema"] == 1
    assert result["per_slot"]["edema"]["informative"] is False
    assert result["per_slot"]["hemisphere"]["informative"] is True
