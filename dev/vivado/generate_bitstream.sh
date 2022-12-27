#!/bin/bash
cd `dirname $0`
proj_name="dvmvs"
rm -rf ${proj_name} || true
vivado -mode batch -source ${proj_name}.tcl
if [ $? != 0 ]; then
    echo "Failed to generate bitstream." 1>&2
    exit 1
fi
dst_dir="design_1"
rm -rf ${dst_dir} || true
mkdir ${dst_dir}
cp ${proj_name}/${proj_name}.runs/impl_1/design_1_wrapper.bit ${dst_dir}/design_1.bit
cp ${proj_name}/${proj_name}.gen/sources_1/bd/design_1/hw_handoff/design_1.hwh ${dst_dir}/design_1.hwh
current_dir=`pwd`
echo "Bitstream is saved in ${current_dir}/design_1."