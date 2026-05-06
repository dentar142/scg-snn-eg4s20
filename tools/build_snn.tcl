# build_snn.tcl - synth scg_top_snn (256→64→3 LIF SNN, INT8 weights baked in)
set proj      scg_top_snn
set top       scg_top_snn
set device_db eagle_s20.db
set package   EG4S20BG256
set speed_grade_candidates [list "" 4 5 6 8 "C7"]

set rtl_files {
    rtl/scg_top_snn.v
    rtl/scg_snn_engine.v
}
set adc_file constraints/scg_top.adc

file mkdir build_snn
cd build_snn

set imported 0
foreach sg $speed_grade_candidates {
    set cmd [list import_device $device_db -package $package]
    if {$sg ne ""} { lappend cmd -speed $sg }
    if {[catch [list eval $cmd] err]} {
        puts "import_device with speed=\"$sg\" failed: $err"
    } else {
        puts "import_device succeeded with speed=\"$sg\""
        set imported 1
        break
    }
}
if {!$imported} { error "Could not import EG4S20BG256 with any candidate speed grade" }

set_param flow qor_monitor on
set_param rtl reg_threshold 0
set_param gate pack_seq_in_io off
set_param place pr_strategy 1

set rtl_paths [list "../rtl/scg_top_snn.v" "../rtl/scg_snn_engine.v"]
puts "read_hdl files: $rtl_paths"
read_hdl -file $rtl_paths -top $top

read_adc ../$adc_file

puts "==> optimize_rtl"
optimize_rtl
report_area              -file ${proj}_rtl.area

puts "==> optimize_gate"
optimize_gate
report_qor  -step gate -file ${proj}_gate.qor
report_area              -file ${proj}_gate.area

puts "==> legalize_phy_inst"
legalize_phy_inst
export_db ./${proj}_gate.db

puts "==> place"
place
update_timing -mode manhattan
report_qor            -step place -file ${proj}_place.qor
report_area                       -file ${proj}_place.area
report_timing_summary             -file ${proj}_place_timing.rpt

puts "==> route"
route
update_timing -mode final
export_db ./${proj}_route.db
report_qor            -step route -file ${proj}_route.qor
report_area                       -file ${proj}_route.area
report_timing_summary             -file ${proj}_route_timing.rpt

puts "==> bitgen"
bitgen -bit ./${proj}.bit

puts "============================================================"
puts " SNN Build complete"
puts " bitstream: build_snn/${proj}.bit"
puts "============================================================"
exit 0
