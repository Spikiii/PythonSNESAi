-- SNES Ai lua script part
-- This just gets ram data, puts in inputs, and interacts with the python script.
-- The python script is the 'brains' of the program.

print("\nStart Script")
local csv = require "simplecsv"

--Declarations
buttonNames = {"A", "B", "X", "Y", "Up", "Down", "Left", "Right"}
frame = 1

--Settings
filepath = "ram.csv"

function getRamValue(add)
	return memory.readbyte(add)
end

function clearJoypad()
	controller = {}
	for b = 1, #buttonNames do
		controller["P1 " .. buttonNames[b]] = false
	end
	joypad.set(controller)
end

function inp(keys, frames)
	controller = {}
	for i = 1, frames do
		for b = 1, #keys do
			controller["P1 " .. keys[b]] = true
			joypad.set(controller)
			emu.frameadvance()
		end
	end
	clearJoypad()
end

function getBytes(startpoint, endpoint)
	bytes = {}
	for i = 0, (endpoint - startpoint) do
		bytes[i + 1] = startpoint + i
	end
	return bytes
end

function writeRamValues(readBytes)
	local f = csv.read(filepath)
	vals = {}
	for b = 1, #readBytes do
		vals[b] = getRamValue(readBytes[b])
	end
	table.insert(f, vals)
	return f
end

function clearRamValues()
	local f = csv.read(filepath)
	simplecsv.write(filepath, {})
end

clearRamValues()

--0x7E0000:0x7FFF00}

vals = {}

for i = 1, 100 do
	table.insert(vals, writeRamValues(getBytes(0x7E0000, 0x7FFF00)))
	frame = frame + 1
	emu.frameadvance()
end

local f = csv.read(filepath)

simplecsv.write(filepath, vals)

print("End Script")