docker run --name isaac-sim-oige --entrypoint bash -it -d --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
-v /etc/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json \
-v /etc/vulkan/implicit_layer.d/nvidia_layers.json:/etc/vulkan/implicit_layer.d/nvidia_layers.json \
-v ${PWD}:/workspace/omniisaacgymenvs \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/config:/root/.nvidia-omniverse/config:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
nvcr.io/nvidia/isaac-sim:2022.2.1

docker exec -it isaac-sim-oige sh -c "cd /workspace/omniisaacgymenvs && /isaac-sim/python.sh -m pip install -e . && cd omniisaacgymenvs"
docker exec -it -w /workspace/omniisaacgymenvs/omniisaacgymenvs isaac-sim-oige bash
