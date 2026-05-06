# download_sram.tcl - flash bitstream to FPGA SRAM (volatile, lost on power-off)
# Use this to fastest-validate the build. For persistent flash, use program_spi.
puts "==> read_device_id (sanity)"
catch {read_device_id} did
puts "  $did"
puts ""
puts "==> download -mode sram -bit build/scg_top.bit"
if {[catch {download -mode sram -bit build/scg_top.bit} err]} {
    puts "ERROR: $err"
    # try without explicit mode
    puts ""
    puts "==> retry: download -bit build/scg_top.bit"
    if {[catch {download -bit build/scg_top.bit} err2]} {
        puts "ERROR: $err2"
    }
}
puts ""
puts "==> done"
exit
