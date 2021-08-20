## Quick Start

#### 1. define custom Model

```python
import ever as er

# register this model to global environment with a name of 'Custom'
@er.registry.MODEL.register()
class Custom(er.ERModule):
    def __init__(self, config):
        super(Custom,self).__init__(config)
    
    def forward(self, *input):
        # todo: coding your logic
        pass
    
    def set_default_config(self):
        self.config.update(dict(
        ))
```

#### 2. define custom Dataloader

```python
import ever as er
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.nums = list(range(10))
    def __getitem__(self, idx):
        return self.nums[idx]
    def __len__(self):
        return len(self.nums)

# register this dataloder to global environment with a name of 'CustomDataloader'
@er.registry.DATALOADER.register()
class CustomDataloader(er.ERDataLoader):
    def __init__(self, config):
        super(CustomDataloader,self).__init__(config)
    
    # override this property to implement your custom logic
    @property
    def dataloader_params(self):
        cfg = self.config
        # todo: instantiate dataset, sampler, ..., etc.
        return dict(
            dataset=CustomDataset(),
            batch_size=1,
            shuffle=cfg.shuffle,
            sampler=None,
            batch_sampler=None,
            num_workers=0
        )
    
    def set_default_config(self):
        self.config.update(dict(
            shuffle=False
        ))
```

#### 3. Create config file

```python
config = dict(
    model=dict(
        type='Custom',
        params=dict(
            # set parameters here 
        )
    ),
    data=dict(
        train=dict(
            type='CustomDataloader',
            params=dict(
                # set parameters here 
            )
        ),
        test=dict(
            type='CustomDataloader',
            params=dict(
                # set parameters here 
            )
        )
    ),
    optimizer=dict(
        type='sgd',
        params=dict(
            momentum=0.9,
            weight_decay=0.001
        )
    ),
    learning_rate=dict(
        type='poly',
        params=dict(
            base_lr=0.001,
            power=0.9,
            max_iters=1000),
    ),
    train=dict(
        # set train config here 
    ),
    test=dict(
    ),
)


```