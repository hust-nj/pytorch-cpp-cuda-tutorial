# pytorch-cpp-cuda tutorial
This is the code of tutorial about how to use c++ and cuda to accelerate the code of forward and backward in pytorch from <https://pytorch.org/tutorials/advanced/cpp_extension.html>

## Installation
```shell
bash init.sh
```

## Compare the speed
```shell
python -m tools.compare
```
to compare the speed of 
- naive pytorch code
- c++ pytorch code
- c++ with cuda kernel pytorch code
