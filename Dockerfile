FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
WORKDIR /evaluate_gender_bias
COPY . .
ENV CUDA_HOME=/usr/local/cuda
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ARG DEBIAN_FRONTEND=noninteractive
RUN chmod +x ./docker/install_python_requirements.sh \
    && chmod +x ./docker/install_r.sh
RUN ./docker/install_python_requirements.sh
RUN ./docker/install_r.sh
RUN Rscript ./docker/install_r_packages.R

# Set the entrypoint to a script that handles input arguments
ENTRYPOINT ["/evaluate_gender_bias/docker/entrypoint.sh"]
CMD ["bash"]