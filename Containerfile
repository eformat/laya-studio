# build
FROM registry.access.redhat.com/ubi9/python-312:latest AS builder
# Dependencies first: only requirements.txt invalidates the install and the
# ~2-3 GB checkpoint layers below, so app edits rebuild without re-downloading
# the laya weights.
USER 0
COPY requirements.txt /tmp/src/requirements.txt
RUN /usr/bin/fix-permissions /tmp/src
USER 1001
# Install the application dependencies (s2i assemble runs pip install -r requirements.txt)
RUN /usr/libexec/s2i/assemble
# Bake the laya checkpoints into the image (modelcar pattern): pods start
# instantly with no Hugging Face egress and no cache volume. ~2-3 GB on top
# of the base image. Cached until requirements.txt changes.
ENV HF_HOME=/opt/app-root/hf-cache
RUN python -c "from laya import Router; Router(preload=True)"
# Application sources last: a code edit rebuilds from here on only. The copy
# replicates what s2i assemble does with /tmp/src, minus the already-satisfied
# pip install.
USER 0
ADD . /tmp/src
RUN /usr/bin/fix-permissions /tmp/src \
    && cp -Rf /tmp/src/. /opt/app-root/src/ \
    && /usr/bin/fix-permissions /opt/app-root -P
USER 1001

# deploy
FROM registry.access.redhat.com/ubi9/python-312-minimal:latest
# gcc + python3.12-devel are required at runtime: Triton JIT-compiles its CUDA
# driver stub with a C compiler on the first GPU kernel launch, and that stub
# includes <Python.h>. The minimal UBI Python image ships neither, so GPU
# inference fails with "Failed to find C compiler" / "Python.h: No such file"
# without this.
USER 0
RUN microdnf install -y gcc python3.12-devel \
    && microdnf clean all \
    && rm -rf /var/cache/dnf
USER 1001
# Copy app sources together with the whole virtual environment from the builder image
COPY --from=builder /opt/app-root /opt/app-root
USER 1001
WORKDIR /opt/app-root/src
EXPOSE 8080
# The checkpoints ship in the image: cache-only, no egress at runtime.
# LAYA_DEVICE is left to the environment (the Helm chart pins cpu or cuda).
ENV HF_HOME=/opt/app-root/hf-cache \
    HF_HUB_OFFLINE=1 \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8080
CMD ["python", "-m", "studio.app"]
