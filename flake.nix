{
  description = "nnue-pytorch CUDA development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        cudaPackages = pkgs.cudaPackages_12;

        # Use GCC 13 stdenv for CUDA compatibility
        stdenv = pkgs.gcc13Stdenv;
      in {
        devShells.default = pkgs.mkShell.override { inherit stdenv; } {
          buildInputs = with pkgs; [
            python3
            uv

            rustc
            cargo
            maturin

            cudaPackages.cudatoolkit
            cudaPackages.cuda_nvrtc
            cudaPackages.cuda_cudart
            cudaPackages.cudnn
            cudaPackages.libcublas
            cudaPackages.libcufft
            cudaPackages.libcurand
            cudaPackages.libcusparse
            cudaPackages.libcusolver
            cudaPackages.nccl

            cmake
            pkg-config

            cacert
            openssl
          ];

          shellHook = ''
            export CUDA_PATH=${cudaPackages.cudatoolkit}
            export CUDA_HOME=$CUDA_PATH
            export CUDA_ROOT=$CUDA_PATH
            export CC=${stdenv.cc}/bin/gcc
            export CXX=${stdenv.cc}/bin/g++
            export CUDAHOSTCXX=${stdenv.cc}/bin/g++
            export NVCC_CCBIN=${stdenv.cc}/bin
            export NVCC_PREPEND_FLAGS="-I$CUDA_PATH/include -L$CUDA_PATH/lib"

            if [ -d "/run/opengl-driver/lib" ]; then
              export LD_LIBRARY_PATH=$CUDA_PATH/lib:${cudaPackages.cuda_nvrtc.lib}/lib:/run/opengl-driver/lib:${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
              export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib
            else
              export LD_LIBRARY_PATH=$CUDA_PATH/lib:${cudaPackages.cuda_nvrtc.lib}/lib:${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
              export TRITON_LIBCUDA_PATH=$CUDA_PATH/lib
            fi

            export PATH=$CUDA_PATH/bin:$PATH

            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            export NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt

            echo "nnue-pytorch CUDA dev shell loaded"
          '';
        };
      }
    );
}
