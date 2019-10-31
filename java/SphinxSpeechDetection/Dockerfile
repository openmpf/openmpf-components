ARG BUILD_REGISTRY
ARG BUILD_TAG=latest
FROM ${BUILD_REGISTRY}openmpf_build:${BUILD_TAG} as openmpf_build


FROM ${BUILD_REGISTRY}openmpf_java_component_build:${BUILD_TAG} as build_component

COPY --from=openmpf_build /home/mpf/openmpf-projects/openmpf-build-tools/mpf-maven-deps.tar.gz /root/mpf-maven-deps.tar.gz

RUN tar --extract --file /root/mpf-maven-deps.tar.gz --directory /root/.m2/repository/

COPY pom.xml pom.xml

RUN mvn org.apache.maven.plugins:maven-dependency-plugin:3.1.1:go-offline;

COPY . .

ARG RUN_TESTS=false

RUN if [ "${RUN_TESTS,,}" == true ]; then \
        mvn package -Dmpf.assembly.format=dir; \
    else \
        mvn package -Dmpf.assembly.format=dir -Dmaven.test.skip=true; \
    fi



FROM ${BUILD_REGISTRY}openmpf_java_executor:${BUILD_TAG} as install_component


COPY --from=build_component /home/mpf/component_src/target/plugin-packages/SphinxSpeechDetection/SphinxSpeechDetection \
                            $INSTALL_DIR

ARG BUILD_SHAS
LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Sphinx Speech Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-ref=$BUILD_SHAS \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
