\ Model assing_problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  9354 Z_00 + 7359 Z_01 + 8665 Z_02 + 10049 Z_03 + 10233 Z_10 + 7217 Z_11
   + 7041 Z_12 + 13685 Z_13 + 2830 Z_20 + 8951 Z_21 + 7783 Z_22 + 5484 Z_23
   + 8017 Z_30 + 6044 Z_31 + 10255 Z_32 + 9432 Z_33 + 2400 Y_0 + 3500 Y_1
   + 2600 Y_2 + 4300 Y_3 + 78327 Constant
Subject To
 R0: = 0
 R1: = 0
 R2: = 0
 R3: = 0
 R4: = 0
 R5: = 0
 R6: = 0
 R7: <= 345
 R8: <= 315
 R9: <= 347
 R10: <= 319
 R11: <= 273
 R12: <= 332
 R13: <= 267
 R14: <= 281
 R15: <= 277
 R16: <= 265
 R17: <= 305
 R18: <= 327
 R19: <= 366
 R20: <= 310
 R21: <= 273
 R22: <= 358
 R23: <= 334
 R24: <= 273
 R25: <= 304
 R26: <= 269
 R27: <= 265
 R28: <= 264
 R29: <= 306
 R30: <= 273
 R31: <= 323
 R32: <= 287
 R33: <= 255
 R34: <= 258
 R35: <= 12326
 R36: <= 11353
 R37: <= 5735
 R38: <= 11337
 R39: <= 5660
 R40: <= 4508
 R41: <= 3913
 R42: Y_0 + Y_1 + Y_2 + Y_3 >= 1
 R43: Y_0 + Y_1 + Y_2 + Y_3 <= 2
 R44: Z_00 + Z_10 + Z_20 + Z_30 = 1
 R45: Z_01 + Z_11 + Z_21 + Z_31 = 1
 R46: Z_02 + Z_12 + Z_22 + Z_32 = 1
 R47: Z_03 + Z_13 + Z_23 + Z_33 = 1
 R48: - Z_00 - Z_01 - Z_02 - Z_03 + 2 Y_0 <= 0
 R49: - Z_10 - Z_11 - Z_12 - Z_13 + 2 Y_1 <= 0
 R50: - Z_20 - Z_21 - Z_22 - Z_23 + 2 Y_2 <= 0
 R51: - Z_30 - Z_31 - Z_32 - Z_33 + 2 Y_3 <= 0
 R52: Z_00 + Z_01 + Z_02 + Z_03 - 4 Y_0 <= 0
 R53: Z_10 + Z_11 + Z_12 + Z_13 - 4 Y_1 <= 0
 R54: Z_20 + Z_21 + Z_22 + Z_23 - 4 Y_2 <= 0
 R55: Z_30 + Z_31 + Z_32 + Z_33 - 4 Y_3 <= 0
 R56: 72 Z_00 + 83 Z_01 + 83 Z_02 + 114 Z_03 <= 600
 R57: 74 Z_10 + 54 Z_11 + 71 Z_12 + 94 Z_13 <= 400
 R58: 38 Z_20 + 62 Z_21 + 51 Z_22 + 67 Z_23 <= 700
 R59: 65 Z_30 + 74 Z_31 + 77 Z_32 + 89 Z_33 <= 800
 R60: - Z_00 - 2 Z_10 - 7 Z_20 - 3 Z_30 + 8 Y_0 + 15 Y_1 + 28 Y_2 + 21 Y_3
   <= 22
 R61: - 2 Z_01 - 3 Z_11 - 5 Z_21 - 4 Z_31 + 8 Y_0 + 15 Y_1 + 28 Y_2
   + 21 Y_3 <= 26
 R62: - 3 Z_02 - 5 Z_12 - 3 Z_22 - Z_32 + 8 Y_0 + 15 Y_1 + 28 Y_2 + 21 Y_3
   <= 32
 R63: - 2 Z_03 - 4 Z_13 - 6 Z_23 - 5 Z_33 + 8 Y_0 + 15 Y_1 + 28 Y_2
   + 21 Y_3 <= 31
 R64: <= 246
 R65: <= 270
 R66: <= 277
 R67: <= 361
Bounds
 Constant = 1
Binaries
 Z_00 Z_01 Z_02 Z_03 Z_10 Z_11 Z_12 Z_13 Z_20 Z_21 Z_22 Z_23 Z_30 Z_31 Z_32
 Z_33 Y_0 Y_1 Y_2 Y_3
End
