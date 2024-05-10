# 本地跑通教程

1、代码

拉取 git clone

依赖 pip install -r requirements.txt

2、数据集

补充 MovieRecbk/sysMovie/MFMGNN_IPTVRS/Data/ [未上传]

补充 MovieRecbk/sysMovie/MFMGNN_IPTVRS/savemodel/ [未上传]

3、数据库

利用 iptv_data_from_computer1.sql 文件导入数据 [未上传] 

在 MovieRecbk/MovieRecbk/settings.py 修改本地mysql数据库信息

```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'HOST': "127.0.0.1",
        'PORT': 3306,
        'NAME': "iptv_rs",                       
        'USER': "root",
        'PASSWORD': "1234",                      # 修改数据库密码
    }
}
```

4、运行

前端 IPTV-movieRS/ 下运行 npm run dev

后端 MovieRecbk/ 下运行 python manage.py runserver


注: 推荐页面的功能得部署到服务器才可以使用