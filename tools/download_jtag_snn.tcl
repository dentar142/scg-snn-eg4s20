puts "==> read_device_id"
read_device_id
puts ""
puts "==> download SNN bit"
download -bit build_snn/scg_top_snn.bit -mode jtag -spd 7 -cable 0
puts ""
puts "==> verify"
read_device_id
exit
