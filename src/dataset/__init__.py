from .base import NBRDatasetBase
from .dunnhumby import DunnhumbyDataset
from .instacart import InstacartDataset
from .tafeng import TafengDataset


DATASETS = {
    "dunnhumby": DunnhumbyDataset,
    "instacart": InstacartDataset,
    "tafeng": TafengDataset,
}
