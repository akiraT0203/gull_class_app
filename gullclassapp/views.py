from django.shortcuts import render, redirect
from .forms import ImageForm, LoginForm, SignUpForm
from .models import ModelFile

from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.models import resnet50
import pytorch_lightning as pl

from model import classification_gull
from .models import ModelFile
from PIL import Image

@login_required
def index(request):
    return render(request, 'gullclassapp/index.html')

@login_required
def image_upload(request):
    # 画像がPOSTで送信されたら、フォームの内容を変数に格納
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        # 取得したデータを検証し、問題なければ保存
        if form.is_valid():
            form.save()
            # 取得した画像のパスをテンプレート側に受け渡す
            data = ModelFile.objects.get(id=ModelFile.objects.latest('id').id)
            img_name = request.FILES['image']
            img_url = 'media/documents/{}'.format(img_name) # 画像のパスを定義
            
            # 推論パート
            image = Image.open(img_url).convert('RGB')
            x = classification_gull.transform(image)
            x = x.unsqueeze(0)
            device = torch.device('cpu')
            # モデルのインスタンス化
            net = classification_gull.Net().to(device).eval()
            # パラメータの読み込み
            net.load_state_dict(
                torch.load(
                    'model/classification_gull.pt',  # パラメータを指定
                    map_location=device))
            # 推論、予測値の計算
            with torch.no_grad():
                y = net(x)
                # 正解ラベルを抽出
                y_arg = torch.argmax(y, dim=1)
                # tensor => numpy 型に変換
                y_arg = y_arg.detach().numpy()
                # ラベルの設定
                if y_arg == 0:
                    y_label = 'MINE MINE'
                elif y_arg == 1:
                    y_label = 'スカットル'
                else:
                    y_label = 'ティッピーブルー'
                
                y_proba = F.softmax(y)
                y_proba = y_proba.detach().numpy() #Tensor型→numpy型に変換
                y_proba = y_proba.squeeze() #リストの次元を削除
                y_proba = max(y_proba)
                y_proba = y_proba * 100  # 予測確率を*100
            
            return render(request, 'gullclassapp/classify.html',
                          {'y_label': y_label, 'y_proba':round(y_proba, 2), 'img_url': img_url})
    else:
        form = ImageForm()
        return render(request, 'gullclassapp/index.html', {'form':form})
        
@login_required
def classify(request):
    template_name = 'gullclassapp/classify.html'

# ログインページ
class Login(LoginView):
    form_class = LoginForm
    template_name = 'gullclassapp/login.html'

# ログアウトページ
class Logout(LogoutView):
    template_name = 'gullclassapp/base.html'

# サインアップページ
def signup(request):
  if request.method == 'POST':
    form = SignUpForm(request.POST)
    if form.is_valid():
      form.save()
      #フォームから'username'を読み取る
      username = form.cleaned_data.get('username')
      #フォームから'password1'を読み取る
      password = form.cleaned_data.get('password1')
      # 読み取った情報をログインに使用する情報として new_user に格納
      new_user = authenticate(username=username, password=password)
      if new_user is not None:
         # new_user の情報からログイン処理を行う
        login(request, new_user)
        # ログイン後のリダイレクト処理
      return redirect('index')
  # POST で送信がなかった場合の処理
  else:
    form = SignUpForm()
    return render(request, 'gullclassapp/signup.html', {'form': form})