## 数据API

本产品提供了金融数据api，方便用户调用接口获取各种数据，通过python的api调用接口，返回DataFrame格式的数据和消息，以下是用法

### 导入接口

在python程序里面导入module，然后用注册的用户帐号登录就可以使用行情和参考数据的接口来获取数据了

#### 引入模块

```python
from jaqs.data.dataapi import DataApi
```
#### 登录数据服务器
```python
api = DataApi(addr='tcp://data.tushare.org:8910')
api.login("phone", "token") 
```

### 调用数据接口

主要数据主要分为两大类：

- **市场数据**，目前可使用的数据包括日线，分钟线，实时行情等。
- **参考数据**，包括财务数据、公司行为数据、指数成份数据等。

数据API使用方法参考[DataApi快速入门](https://github.com/quantOS-org/DataCore/blob/master/doc/api_ref.md)



