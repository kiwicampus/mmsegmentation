from mmseg.registry import DATASETS

from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class KiwiScapesDataset(BaseSegDataset):
    """Kiwiscapes dataset.

    A wrapper for the Kiwi Semantic Segmentation dataset. The dataset is
    expected to have the following directory structure:

    kiwiscapes/
    ├── images/
    │   ├── test/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   │
    │   └── train/
    │       ├── image1.jpg
    │       ├── image2.jpg
    │       └── ...
    └── labels/
        ├── test/
        │   ├── label1.png
        │   ├── label2.png
        │   └── ...
        └── train/
            ├── label1.png
            ├── label2.png
            └── ...

    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '_seg.png' for Kiwiscapes dataset.
    """

    METAINFO = dict(
        classes=("background", "drivable", "off-road", "sidewalk-step", "car-street"),
        palette=[
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
        ],
    )

    def __init__(
        self,
        img_suffix: str = ".jpg",
        seg_map_suffix: str = "_seg.png",
        **kwargs: dict,
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=True,
            ignore_index=0,
            **kwargs,
        )
