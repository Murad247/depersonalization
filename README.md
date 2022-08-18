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
```
Далее скопируйте всё необходимое в одну директорию:
```
cp -r .\depersonalization\* .\yolov7
cp -r .\participants\ .\yolov7
```
