"""probe_uart.py - cheap diagnostics over UART without expecting protocol."""
import serial, time, sys
port = sys.argv[1] if len(sys.argv) > 1 else "COM27"
ser = serial.Serial(port, 115200, timeout=1.0)
time.sleep(0.1)

def hex_dump(name, data):
    print(f"{name}: {len(data)} B  {' '.join(f'{b:02X}' for b in data[:16])}{'…' if len(data)>16 else ''}")

# Drain any prior bytes
ser.reset_input_buffer()

# Test 1: cold CMD_RUN — should engine still respond? Engine reads zeros (BRAM init)
ser.write(b"\xA3"); ser.flush()
r = ser.read(4)
hex_dump("cold CMD_RUN reply", r)

# Test 2: CMD_RST then CMD_RUN
ser.write(b"\xA0"); ser.flush(); time.sleep(0.01)
ser.reset_input_buffer()
ser.write(b"\xA3"); ser.flush()
r = ser.read(4)
hex_dump("RST + RUN reply", r)

# Test 3: send unknown 0x55 byte — FSM stays IDLE, nothing returned
ser.write(b"\x55"); ser.flush()
r = ser.read(4)
hex_dump("0x55 reply (should be empty)", r)

# Test 4: full 256B window of zeros + RUN
ser.reset_input_buffer()
ser.write(b"\xA0"); ser.flush(); time.sleep(0.01)
ser.write(b"\xA2" + b"\x00" * 256); ser.flush()
time.sleep(0.005)
ser.write(b"\xA3"); ser.flush()
r = ser.read(4)
hex_dump("Window=zeros + RUN", r)

ser.close()
