from django.db import models


class User_table(models.Model):
    username = models.CharField(max_length=20, verbose_name='用户名')
    password = models.CharField(max_length=20, verbose_name='密码')
    id = models.IntegerField(verbose_name='用户id',primary_key=True)
    token = models.CharField(max_length=255, null=True)
    email = models.CharField(max_length=40, null=True,verbose_name='邮箱')
    phoneNumber = models.CharField(max_length=11, null=True,verbose_name='电话')
    sex = models.CharField(max_length=5, null=True,verbose_name='性别')

    def __str__(self):
        return self.username

    class Meta:
        db_table = 'rs_user'
        verbose_name_plural = "用户信息"


class Historyzhanshi_table(models.Model):
    name = models.CharField(blank=False, max_length=20, verbose_name='用户名')
    uid = models.IntegerField(blank=False,verbose_name='用户id')
    item_name = models.CharField(blank=False,max_length=100,verbose_name='电影名称')
    bastags = models.CharField(blank=True, max_length=60, verbose_name='标签')
    action = models.CharField(blank=False, max_length=5, verbose_name='行为')
    time = models.DateTimeField(blank=True, default="2008-12-12 12:12:00", verbose_name="点击时间")
    id = models.IntegerField(blank=True, verbose_name="id",primary_key=True)

    def __str__(self):
        return self.name

    class Meta:
        db_table = 'historyzhanshi'
        verbose_name_plural = "行为信息"