# Build stage: Alpine's Rust toolchain targets musl, and Rust's musl
# targets link statically by default — which is what lets the runtime
# stage be `scratch`. No OpenSSL and no CA bundle are needed: TLS is
# rustls with the webpki roots compiled into the binary.
FROM rust:1.96-alpine AS build
RUN apk add --no-cache musl-dev
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# _archive/docs: llm-protocol.md is include_str!'d into the binary.
COPY _archive/docs ./_archive/docs
RUN cargo build --release --locked --bin taguru \
    && mkdir /data-skeleton

# Runtime stage: the binary is the image — no shell, no libc, no
# package database, nothing to patch or scan.
FROM scratch

LABEL org.opencontainers.image.source="https://github.com/t0k0sh1/taguru" \
      org.opencontainers.image.description="Long-term semantic memory for LLMs" \
      org.opencontainers.image.licenses="MIT"

COPY --from=build /src/target/release/taguru /taguru
# /data must pre-exist owned by the runtime user: scratch has no mkdir,
# and uid 65532 cannot create directories under /. A named volume
# inherits this ownership on first use; a bind mount needs its own
# `chown -R 65532` (or run with --user).
COPY --from=build --chown=65532:65532 /data-skeleton /data

# Loopback would be unreachable through -p from outside the container's
# network namespace; bind everything and let -p choose the exposure.
# UNAUTHENTICATED mode therefore reaches as far as the port is
# published: keep -p scoped to 127.0.0.1, or set TAGURU_API_TOKEN.
ENV TAGURU_ADDR=0.0.0.0:8248 \
    TAGURU_DATA_DIR=/data

USER 65532:65532
EXPOSE 8248
# The binary probes itself — scratch has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=2s \
    CMD ["/taguru", "health"]
ENTRYPOINT ["/taguru"]
