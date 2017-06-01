-- SNES Ai lua script part
-- This just gets ram data, puts in inputs, and interacts with the python script.
-- The python script is the 'brains' of the program.

print("\nStart Script")
local csv = require "simplecsv"

--Declarations
buttonNames = {"A", "B", "X", "Y", "Up", "Down", "Left", "Right"}

--Settings
record = false
rfilepath = "ram.csv"
ifilepath = "inp.csv"
sram = {0x0014A2, 0x001496, 0x000E32, 0x000E33, 0x000E34, 0x000ED3, 0x000ED2, 0x000ED4,
		0x000EE4, 0x000EE2, 0x000EE3, 0x000EF4, 0x000EF2, 0x000E43, 0x000EB2, 0x000EB3,
		0x000EB4, 0x000324, 0x000E44, 0x00031C, 0x000E42, 0x0002A2, 0x0002A6, 0x0002AA,
		0x7E0F34} --This is the ram for the computer to search for, insert this manually

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

function getInputs()
	local i = csv.read(ifilepath)
	inpOrder = {"Up", "Down", "Left", "Right", "A", "B", "X", "Y", "Lb", "Rb"}
	inputs = {}
	for j = 1, #i do
		if i[j] == 1 then
			inputs[#inputs + 1] = inpOrder[j]
		end
	end
	inp(inputs, 1)
end

function updateRam()
	score = getRamValue(0x7E0F36) * 655360 + getRamValue(0x7E0F35)  * 2560 + getRamValue(0x7E0F34) * 10	
	b = {}
	f = {}
	
	b = getBytes(0x7E0000, 0x7E1FFF)
	for k = 1, #b do
		b[k] = getRamValue(b[k])
	end
	
	b[#b + 1] = score
	table.insert(f, b)
	csv.write(rfilepath, f)
end


function compareTables(a, b)
	--Used for comparing inputs...
	equal = true
	for i = 1, #a do
		if(a[i] ~= b[i] and equal == true) then
			equal = false
		end
	end
	return equal
end
	
--0x7E0000:0x7E1FFF

if record then
	local f = {}
	for j = 1, 500 do
		
		score = getRamValue(0x7E0F36) * 655360 + getRamValue(0x7E0F35)  * 2560 + getRamValue(0x7E0F34) * 10
		b = getBytes(0x7E0000, 0x7E1FFF)
		for k = 1, #b do
			b[k] = getRamValue(b[k])
		end
		
		b[#b + 1] = score
		table.insert(f, b)
		
		frame = frame + 1
		emu.frameadvance()
		emu.frameadvance()
		emu.frameadvance()
		emu.frameadvance()
		
	end
	csv.write("testRam.csv", f)
	
else
	previ = csv.read(ifilepath)
	for q = 1, 100000 do
	
		local i = csv.read(ifilepath)
		print("Searching...")
		if(compareTables(previ, i) ~= true) then
			print("Update!")
			getInputs()
			updateRam()
			emu.frameadvance()
			previ = i
			frame = frame + 1
		end
		
		if q == 1 then
			updateRam()
		end
		
	end
end
			
print("End Script")