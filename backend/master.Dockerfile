# Base image: plain Ubuntu
FROM ubuntu:22.04

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ARG GMTSAR=/usr/local/GMTSAR
ARG ORBITS=/usr/local/orbits
ENV PATH=${GMTSAR}/bin:$PATH

# Create user (instead of NB_USER)
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# System packages and Python setup
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    gfortran \
    make \
    autoconf \
    subversion \
    csh \
    gdal-bin \
    libgdal-dev \
    libtiff5-dev \
    libhdf5-dev \
    liblapack-dev \
    libgmt-dev \
    gmt \
    zip \
    htop \
    mc \
    netcdf-bin \
    xvfb \
    libegl1-mesa \
    jq \
    pkg-config \
    libproj-dev \
    libgeos-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user and grant passwordless sudo
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install GMTSAR
RUN cd $(dirname ${GMTSAR}) \
    && git config --global advice.detachedHead false \
    && git clone --branch master https://github.com/gmtsar/gmtsar GMTSAR \
    && cd ${GMTSAR} \
    && git checkout e98ebc0f4164939a4780b1534bac186924d7c998 \
    && autoconf \
    && ./configure --with-orbits-dir=${ORBITS} CFLAGS='-z muldefs' LDFLAGS='-z muldefs' \
    && make \
    && make install

# Install Python dependencies in specific order to avoid conflicts
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first (required by many other packages)
RUN pip3 install --no-cache-dir numpy==1.26.4

# Install core scientific packages
RUN pip3 install --no-cache-dir \
    pandas==2.1.4 \
    xarray==2023.12.0 \
    scipy==1.11.4

# Install geospatial packages - MATCH THE WORKING COLAB VERSIONS
RUN pip3 install --no-cache-dir \
    rasterio==1.3.11 \
    geopandas==1.0.1 \
    shapely==2.0.2 \
    pyproj==3.6.1 \
    fiona==1.9.6

# Install other dependencies
RUN pip3 install --no-cache-dir \
    asf_search==7.0.4 \
    h5netcdf==1.3.0 \
    h5py==3.10.0 \
    ipywidgets==8.1.1 \
    ipyleaflet==0.19.1 \
    remotezip==0.12.2 \
    fastapi \
    uvicorn \
    python-multipart \
    pydantic \
    matplotlib \
    dask[complete] \
    distributed

# Install pygmtsar - should match the working version
RUN pip3 install --no-cache-dir pygmtsar

# Set environment variables for better compatibility
ENV PYGMTSAR_SCHEDULER=synchronous
ENV OMP_NUM_THREADS=4
ENV DASK_DISTRIBUTED__WORKER__DAEMON=False
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages:$PYTHONPATH

# Create data directories
RUN mkdir -p /app/sentinel1_burst_downloads /app/insar_processing /app/insar_results \
    && chown -R $USERNAME:$USERNAME /app

# Copy backend code
WORKDIR /app
COPY *.py /app/
RUN chown -R $USERNAME:$USERNAME /app

# Switch to non-root user
USER $USERNAME

# Add health check
HEALTHCHECK --interval=60s --timeout=20s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Command to run FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]