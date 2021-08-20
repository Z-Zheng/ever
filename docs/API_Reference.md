#### Basic Components
```python
import ever as er
# manager for configure file
er.config
# register for custom dataloader, module
er.registry

# two core components
# ERModule is an interface for model
er.ERModule
# ERDataLoader is an interface for loading data
er.ERDataLoader

# preset modules, network, loss, etc.
import ever.module as erm

# tools
er.metric
# visualize
er.viz
# for train model
er.trainer
# many preprocess methods, e.g., flip, random scale
er.preprocess
# a quick way to build model from config file and checkpoint file
er.infer_tool
# for model statistic
er.param_util
```