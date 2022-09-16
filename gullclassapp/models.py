from distutils.command.upload import upload
from django.db import models

# Create your models here.
class ModelFile(models.Model):
  # DBのカラムに相当する部分の定義
    id = models.AutoField(primary_key=True)
    result = models.IntegerField(blank=True, null=True) #nullを許可する場合、blank=Trueにしないとエラーになる
    image = models.ImageField(upload_to='documents/') # MEDIA_ROOT 配下のdocumentsフォルダを参照
    proba = models.FloatField(default=0.0)
    comment = models.CharField(max_length=200, blank=True, null=True)
