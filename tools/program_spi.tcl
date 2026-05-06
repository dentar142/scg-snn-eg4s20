# program_spi.tcl - persist scg_top.bit to on-board SPI flash via JTAG
# After this, the design survives power cycle (loaded automatically on power-up).
# Reference: SWUG110 BitWizard User Guide §4.3 program_spi.

puts "==> read_device_id"
read_device_id

# Convert .bit -> .vec for SPI flash flow
puts "==> bit_to_vec for SPI mode"
bit_to_vec -chip EG4S20BG256 -m spi -freq 5.000000 -bit build/scg_top.bit

# program_spi: erase + program + verify SPI flash
# -cable 0 -spd 7 = JTAG cable index 0, speed level 7
puts "==> program_spi (erase + program + verify)"
program_spi -cable 0 -spd 7 -mode all -bit build/scg_top.bit

puts "==> done. Power cycle to confirm bitstream loads from SPI flash."
exit
