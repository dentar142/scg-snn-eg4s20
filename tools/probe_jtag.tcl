# probe_jtag.tcl - read device ID through Anlogic USB cable to confirm chain.
# Run with: bw_commands_prompt.exe probe_jtag.tcl
puts "==> list_cable"
catch {list_cable} cab
puts $cab
puts ""
puts "==> read_device_id"
if {[catch {read_device_id} did]} {
    puts "read_device_id failed: $did"
} else {
    puts "device id chain: $did"
}
exit
