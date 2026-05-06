puts "==> info bit file"
set bit_path "build/scg_top.bit"
puts "  bit file: $bit_path  size = [file size $bit_path] bytes"
puts ""
puts "==> read_device_id"
read_device_id
puts ""
puts "==> download -v -mode sram -bit ..."
download -v -mode sram -bit $bit_path
puts ""
puts "==> read_device_id again"
read_device_id
exit
