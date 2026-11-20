# Test fixture only; this image contains no arbiter code or credentials.
FROM busybox:1.37.0

COPY fixture.sh /fixture.sh
RUN chmod 0555 /fixture.sh
USER 65534:65534

ENTRYPOINT ["/bin/sh", "/fixture.sh"]
