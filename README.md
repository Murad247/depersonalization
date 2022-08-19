# depersonalization

Для воспроизведения результатов выполните следующие команды в Anaconda prompt:  
```
conda create -n murad_millinov python=3.8
conda activate murad_millinov
pip install jupyter
jupyter notebook
```

Далее в терминале Jupyter Notebook клонируйте необходимые репозитории:
```
git clone https://github.com/WongKinYiu/yolov7.git
pip install -r ./yolov7/requirements.txt
git clone https://github.com/Murad247/depersonalization.git
```
и установите библиотеки:
```
pip install map-boxes
pip install sklearn
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html --user
pip install setuptools==59.5.0
```

### В директории, в которую вы ранее клонировали два репозитория, должна быть папка participants (с такой же структурой, как в baseline).
Далее скопируйте всё необходимое в одну директорию:
```
cp -r .\depersonalization\* .\yolov7
cp -r .\participants\ .\yolov7
```
После этого запустите блокнот yolov7/train_detect_getSolution.ipynb и следуйте инструкциям в нём
