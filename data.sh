mkdir -p evaluation/data && cd evaluation/data
huggingface-cli download --repo-type dataset Vchitect/ShotBench --local-dir ShotBench
cd ShotBench
tar -xvf images.tar
tar -xvf videos.tar
cd ../../../