# 使用包含 Miniconda 的 Docker 镜像作为基础镜像
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /usr/src/app/handwriting-recognition

# 将当前目录下的所有文件复制到工作目录内
COPY . .

# 使用 environment.yml 创建 Conda 环境
RUN conda env create -f environment.yml

# 激活 Conda 环境
SHELL ["conda", "run", "-n", "handwriting-recognition", "/bin/bash", "-c"]

# 运行你的代码或启动服务，例如：
# CMD ["conda", "run", "-n", "handwriting-recognition", "python", "your_script.py"]
