puts "==> read_device_id"
read_device_id
puts ""
puts "==> download v7 bit"
download -bit build_v7/scg_top_v7.bit -mode jtag -spd 7 -cable 0
puts ""
puts "==> verify"
read_device_id
exit
