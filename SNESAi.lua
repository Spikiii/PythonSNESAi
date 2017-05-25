-- SNES Ai lua script part
-- This just gets ram data, puts in inputs, and interacts with the python script.
-- The python script is the 'brains' of the program.

print("\nStart Script")
local csv = require "simplecsv"

--Declarations
buttonNames = {"A", "B", "X", "Y", "Up", "Down", "Left", "Right"}
frame = 1

--Settings
record = false
rfilepath = "ram.csv"
ifilepath = "inp.csv"
sram = {0x0014A2, 0x001496, 0x000E32, 0x000E33, 0x000E34, 0x000ED3, 0x000ED2, 0x000ED4,
		0x000EE4, 0x000EE2, 0x000EE3, 0x000EF4, 0x000EF2, 0x000E43, 0x000EB2, 0x000EB3,
		0x000EB4, 0x000324, 0x000E44, 0x00031C, 0x000E42, 0x0002A2, 0x0002A6, 0x0002AA} --This is the ram for the computer to search for, insert this manually

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
	csv.write(rfilepath, {""})
end

function getInputs()
	local f = csv.read(ifilepath)
	i = f[frame]
	inpOrder = {"Up", "Down", "Left", "Right", "A", "B", "X", "Y", "Lb", "Rb"}
	inputs = {}
	for j = 1, #i do
		if i[j] == 1 then
			inputs[#inputs + 1] = inpOrder[j]
		end
	end
	inp(inputs, 1)
end
	
--clearRamValues()

--0x7E0000:0x7FFF00

local f = csv.read(rfilepath)

if record then	
	for j = 1, 100 do
		table.insert(f, writeRamValues(getBytes(0x7E0000, 0x7E00FF)))
		frame = frame + 1
		emu.frameadvance()
	end
	csv.write(rfilepath, f)
end

else
	prevFrame = 1
	while true do
		local i = csv.read(ifilepath)
		if #i + 1 > preFrame then
			getInputs()
			
			b = {}
			for j = 1, #sram do
				b[#b + 1] = getRamValue(sram[j])
			end
			
			table.insert(f, b)
			csv.write(rfilepath, f)
			
			emu.frameadvance()
			prevFrame = #i + 1
			frame = frame + 1
		end
	end
end
			
print("End Script")