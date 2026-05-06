# ============================================================================
# build.tcl - Anlogic TD 6.2.x non-project batch flow
#
# Run with:
#   D:\Anlogic\TD_Release_2026.1_6.2.190.657\bin\td_commands_prompt.exe build.tcl
# (or via tools/Makefile target `synth`)
#
# Project root must be the cwd when this script is sourced.
# ============================================================================

set proj      scg_top
set top       scg_top
set device_db eagle_s20.db
set package   EG4S20BG256
# Speed grade: EG4S20 uses {C7, C8, I7} — not the "1-4" used by PH1.
# Try common values until one is accepted; leave blank to use device default.
set speed_grade_candidates [list "" 4 5 6 8 "C7"]

set rtl_files {
    rtl/scg_top.v
    rtl/scg_mac_array.v
}
set adc_file constraints/scg_top.adc

file mkdir build
cd build

# ------------------------------------------------------------ device --------
# Try each candidate speed grade in order; first that succeeds wins.
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
if {!$imported} {
    error "Could not import EG4S20BG256 with any candidate speed grade"
}

# ----------------------------------------------------------- params --------
set_param flow qor_monitor on
set_param rtl reg_threshold 0
set_param gate pack_seq_in_io off
set_param place pr_strategy 1

# ------------------------------------------------------------ HDL ----------
# Build the file list as a brace-quoted Tcl list so TD parses it as a single
# `-file` argument value rather than as two positional tokens.
set rtl_paths [list "../rtl/scg_top.v" "../rtl/scg_mac_array.v"]
puts "read_hdl files: $rtl_paths"
read_hdl -file $rtl_paths -top $top

# ----------------------------------------------------- constraints ---------
read_adc ../$adc_file
# read_sdc ../$sdc_file   ; # add when an SDC is available

# ------------------------------------------------- synthesis flow ----------
# (read_hdl -top already analyzes + elaborates; no separate elaborate call)
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

# ------------------------------------------------- place / route -----------
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

# --------------------------------------------------------- bitgen ----------
puts "==> bitgen"
bitgen -bit ./${proj}.bit

puts ""
puts "============================================================"
puts " Build complete"
puts " bitstream     : build/${proj}.bit"
puts " gate area     : build/${proj}_gate.area"
puts " place timing  : build/${proj}_place_timing.rpt"
puts " route timing  : build/${proj}_route_timing.rpt"
puts "============================================================"
exit 0
