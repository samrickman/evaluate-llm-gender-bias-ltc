# Install requirements

if [[ "$VIRTUAL_ENV" == "" ]]; then 
    echo "Activating virtual env"
    source .venv/bin/activate
fi
apt update
apt install -y git # needed for flash-attn for some reason
python3.10 -m pip install -r ./docker/requirements.txt
