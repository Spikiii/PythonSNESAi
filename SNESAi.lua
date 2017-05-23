-- SNES Ai lua script part
-- This just gets ram data, puts in inputs, and interacts with the python script.
-- The python script is the 'brains' of the program.

print("\nStart Script")
local csv = require "simplecsv"

--Declarations
buttonNames = {"A", "B", "X", "Y", "Up", "Down", "Left", "Right"}
frame = 1

--Settings
record = true
rfilepath = "ram.csv"
ifilepath = "inp.csv"
sram = {} --This is the ram for the computer to search for, insert this manually

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
	local f = csv.read(rfilepath)
	vals = {}
	for b = 1, #readBytes do
		vals[b] = getRamValue(readBytes[b])
	end
	return vals
end

function clearRamValues()
	local f = csv.read(rfilepath)
	simplecsv.write(rfilepath, {})
end

function getInputs()
	local f = csv.read(ifilepath)
	i = f[frame]
	inpOrder = {"Up", "Down", "Left", "Right", "A", "B", "X", "Y", "Lb", "Rb"}
	inputs = {}
	for j = 1, #i do
		if i[j] == 1 do
			inputs[#inputs + 1] = inpOrder[j]
		end
	end
	inp(inputs, 1)
end
	
clearRamValues()

--0x7E0000:0x7FFF00

local f = csv.read(rfilepath)
prevInp = #i

if record do	
	for j = 1, 100 do
		table.insert(f, writeRamValues(getBytes(0x7E0000, 0x7FFF00)))
		frame = frame + 1
		emu.frameadvance()
	end
	simplecsv.write(filepath, f)
end

--This is commented out until I can get it working tomorrow.

--else do
--	while true do
--		local i = csv.read(ifilepath)
--		for b = 1, #sram
--	end
--end

print("End Script")