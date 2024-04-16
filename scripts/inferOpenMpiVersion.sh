cli_arg=$1
OMPI_VERSION_STRING=$cli_arg

if [ -z "$cli_arg" ]; then
    path=$PATH
    OMPI_VERSION_STRING=$([[ $path =~ openmpi-[0-9]+\.[0-9]+\.[0-9]+ ]] && echo $BASH_REMATCH)
fi

if [ -z "$OMPI_VERSION_STRING" ]; then
    echo "Open MPI not found in PATH and no version string provided as argument."
    exit 1
fi

OMPI_VERSION=$([[ $OMPI_VERSION_STRING =~ [0-9]+\.[0-9]+\.[0-9]+ ]] && echo $BASH_REMATCH)

sed 's/OMPI_VERSION=__VERSION__/OMPI_VERSION='$OMPI_VERSION'/g' PQ_openmpi.def
