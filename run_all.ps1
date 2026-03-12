python craft_core.py --nodes 8 --experts 256 --layers 61 --first-moe-layer 3 --dist-file .\traces\DE.pkl
python craft_core.py --nodes 8 --experts 256 --layers 61 --first-moe-layer 3 --dist-file .\traces\DJ.pkl
python craft_core.py --nodes 8 --experts 256 --layers 61 --first-moe-layer 3 --dist-file .\traces\DL.pkl
python craft_core.py --nodes 8 --experts 256 --layers 61 --first-moe-layer 3 --dist-file .\traces\DA.pkl

python craft_core.py --nodes 8 --experts 384 --layers 61 --first-moe-layer 1 --dist-file .\traces\KE.pkl
python craft_core.py --nodes 8 --experts 384 --layers 61 --first-moe-layer 1 --dist-file .\traces\KJ.pkl
python craft_core.py --nodes 8 --experts 384 --layers 61 --first-moe-layer 1 --dist-file .\traces\KL.pkl
python craft_core.py --nodes 8 --experts 384 --layers 61 --first-moe-layer 1 --dist-file .\traces\KA.pkl

python gen_ae_fig.py