# Product Candidates deployment templates

These files describe the approved engineering topology only. They do not
install, enable, or start a service.

## Fixed topology

- Application: `app_product_candidates:app`
- Working directory: `/home/kyzen/ULIP_RAG/ikea`
- Python path: `/home/kyzen/ULIP_RAG/core:/home/kyzen/ULIP_RAG/ikea`
- Bind: `127.0.0.1:8321`
- Uvicorn workers: `1`
- 5090 transport: SSH tunnel to the loopback listener
- Port `8000`: untouched

The ASGI application module is fixed in the launcher and cannot be replaced by
an EnvironmentFile value.  This prevents an operational configuration from
silently selecting a different serving path.

The launcher exports the decision values for a 25-second application request
deadline, inference concurrency `1`, and maximum queue `1`. The application
must enforce those semantics; Uvicorn's keep-alive timeout is not a request
deadline and is intentionally not presented as one.

## EnvironmentFile

`ulip-product-candidates.env.example` is safe to commit because it contains no
credential. Before a future authorized installation, an operator must copy it
outside the repository, replace both placeholders, and make the real file mode
`0600`. The unit template expects it at:

```text
/etc/ulip-product-candidates/environment
```

The bearer value must encode at least 32 random bytes. The launcher performs a
minimum placeholder/length guard without printing the value; the application
remains responsible for cryptographic token validation and constant-time
comparison.

## Systemd boundary

`ulip-product-candidates.service` is a template in the repository, not an installed unit.
Installing it under `/etc/systemd/system`, creating the external EnvironmentFile,
running `daemon-reload`, enabling the unit, or changing firewall rules requires
separate operator authorization. The template only permits IP traffic to
localhost and deliberately does not use `PrivateDevices=true`, because that
would hide the NVIDIA GPU.

`GET /healthz` and `GET /readyz` may be unauthenticated on loopback/tunnel. The
product-candidates POST endpoint must require the bearer token. A conforming
`/readyz` response also carries `Cache-Control: no-store`; that HTTP header is
outside the JSON schema.
