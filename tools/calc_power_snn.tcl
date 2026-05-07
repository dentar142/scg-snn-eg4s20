# calc_power_snn.tcl — Anlogic TD power flow:
#   create_power_report → set_switching_activity → calculate_power → export_*
set proj      scg_top_snn
set device_db eagle_s20.db
set package   EG4S20BG256

cd build_snn
foreach sg [list "" 4 5 6 8 "C7"] {
    set cmd [list import_device $device_db -package $package]
    if {$sg ne ""} { lappend cmd -speed $sg }
    if {![catch [list eval $cmd] err]} { break }
}
import_db ./${proj}_route.db

set rpt ${proj}_power_rpt

puts "==> create_power_report (default toggle 0.125)"
if {[catch {create_power_report -report $rpt -overwrite -defTogg 0.125 -amb 25 -airFlow 0 -junc 25} err]} {
    puts "create_power_report FAILED: $err"
    exit 1
}

puts "==> calculate_power"
if {[catch {calculate_power -report $rpt} err]} {
    puts "calculate_power FAILED: $err"
    exit 2
}

puts "==> generate_text_report"
if {[catch {generate_text_report -report $rpt} err]} {
    puts "generate_text_report FAILED: $err"
}

puts "==> export_power_report"
if {[catch {export_power_report -report $rpt} err]} {
    puts "export_power_report FAILED: $err"
}

puts "==> done"
exit 0
