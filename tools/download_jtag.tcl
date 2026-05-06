# download_jtag.tcl - flash bit to FPGA SRAM via JTAG (volatile)
# Syntax cribbed from SWUG110 BitWizard User Guide example.
puts "==> read_device_id"
read_device_id
puts ""
puts "==> download -bit build/scg_top.bit -mode jtag -spd 7 -cable 0"
download -bit build/scg_top.bit -mode jtag -spd 7 -cable 0
puts ""
puts "==> read_device_id (post-download verify)"
read_device_id
exit
