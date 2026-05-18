# ------------------ 1. 基础镜像 ------------------
FROM continuumio/miniconda3:latest

# ------------------ 2. 设置工作目录 ------------------
WORKDIR /app

# ------------------ 3. 配置 apt 源并安装依赖 ------------------
RUN echo "deb http://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware\n\
deb http://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware\n\
deb http://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware\n\
deb http://mirrors.aliyun.com/debian/ bookworm-backports main contrib non-free non-free-firmware" \
> /etc/apt/sources.list.d/aliyun.list && \
    apt update && \
    apt install -y --no-install-recommends \
        git wget curl   && \
    rm -f /etc/apt/sources.list.d/aliyun.list && \
    apt clean && rm -rf /var/lib/apt/lists/*


COPY environment.yml .
COPY ./app ./app


RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN conda env create -f environment.yml && conda clean -afy

RUN conda run -n embedding pip install --upgrade pip

RUN conda run -n embedding modelscope download --model AI-ModelScope/chinese-clip-vit-huge-patch14 --local_dir ./model

EXPOSE 8002

ENTRYPOINT ["conda", "run", "-n", "embedding", "uvicorn"]

CMD ["app.api_server_embedding:app", "--host", "0.0.0.0", "--port", "8002"]
