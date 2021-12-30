pip install torch==1.4.0 torchvision==0.5.0
pip install git+'https://github.com/otaheri/MANO'
pip install matplotlib==3.3.3

mkdir -p external
git clone https://github.com/JiangWenPL/multiperson.git external/multiperson
pip install external/multiperson/neural_renderer

pip install chumpy