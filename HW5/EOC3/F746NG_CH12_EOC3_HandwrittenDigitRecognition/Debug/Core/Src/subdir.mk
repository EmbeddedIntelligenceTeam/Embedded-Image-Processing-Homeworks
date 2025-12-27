################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../Core/Src/hdr_mlp.cpp \
../Core/Src/lib_model.cpp \
../Core/Src/main.cpp 

C_SRCS += \
../Core/Src/hdr_feature_extraction.c \
../Core/Src/lib_image.c \
../Core/Src/lib_mpu.c \
../Core/Src/lib_ov5640.c \
../Core/Src/lib_rgb2gray.c \
../Core/Src/lib_serial.c \
../Core/Src/lib_serialimage.c \
../Core/Src/lib_slidingwindow.c \
../Core/Src/ov5640.c \
../Core/Src/ov5640_reg.c \
../Core/Src/stm32f7xx_hal_msp.c \
../Core/Src/stm32f7xx_it.c \
../Core/Src/syscalls.c \
../Core/Src/sysmem.c \
../Core/Src/system_stm32f7xx.c 

C_DEPS += \
./Core/Src/hdr_feature_extraction.d \
./Core/Src/lib_image.d \
./Core/Src/lib_mpu.d \
./Core/Src/lib_ov5640.d \
./Core/Src/lib_rgb2gray.d \
./Core/Src/lib_serial.d \
./Core/Src/lib_serialimage.d \
./Core/Src/lib_slidingwindow.d \
./Core/Src/ov5640.d \
./Core/Src/ov5640_reg.d \
./Core/Src/stm32f7xx_hal_msp.d \
./Core/Src/stm32f7xx_it.d \
./Core/Src/syscalls.d \
./Core/Src/sysmem.d \
./Core/Src/system_stm32f7xx.d 

OBJS += \
./Core/Src/hdr_feature_extraction.o \
./Core/Src/hdr_mlp.o \
./Core/Src/lib_image.o \
./Core/Src/lib_model.o \
./Core/Src/lib_mpu.o \
./Core/Src/lib_ov5640.o \
./Core/Src/lib_rgb2gray.o \
./Core/Src/lib_serial.o \
./Core/Src/lib_serialimage.o \
./Core/Src/lib_slidingwindow.o \
./Core/Src/main.o \
./Core/Src/ov5640.o \
./Core/Src/ov5640_reg.o \
./Core/Src/stm32f7xx_hal_msp.o \
./Core/Src/stm32f7xx_it.o \
./Core/Src/syscalls.o \
./Core/Src/sysmem.o \
./Core/Src/system_stm32f7xx.o 

CPP_DEPS += \
./Core/Src/hdr_mlp.d \
./Core/Src/lib_model.d \
./Core/Src/main.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/%.o Core/Src/%.su Core/Src/%.cyclo: ../Core/Src/%.c Core/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m7 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F746xx -DARM_MATH_CM7 -DTF_LITE_STATIC_MEMORY -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/flatbuffers/include" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/gemmlowp" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/kissfft" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/ruy" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Core/Src/%.o Core/Src/%.su Core/Src/%.cyclo: ../Core/Src/%.cpp Core/Src/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m7 -std=gnu++14 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F746xx -DARM_MATH_CM7 -DTF_LITE_STATIC_MEMORY -c -I../Core/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc -I../Drivers/STM32F7xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F7xx/Include -I../Drivers/CMSIS/Include -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/flatbuffers/include" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/gemmlowp" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/kissfft" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/ruy" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition" -O0 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Core-2f-Src

clean-Core-2f-Src:
	-$(RM) ./Core/Src/hdr_feature_extraction.cyclo ./Core/Src/hdr_feature_extraction.d ./Core/Src/hdr_feature_extraction.o ./Core/Src/hdr_feature_extraction.su ./Core/Src/hdr_mlp.cyclo ./Core/Src/hdr_mlp.d ./Core/Src/hdr_mlp.o ./Core/Src/hdr_mlp.su ./Core/Src/lib_image.cyclo ./Core/Src/lib_image.d ./Core/Src/lib_image.o ./Core/Src/lib_image.su ./Core/Src/lib_model.cyclo ./Core/Src/lib_model.d ./Core/Src/lib_model.o ./Core/Src/lib_model.su ./Core/Src/lib_mpu.cyclo ./Core/Src/lib_mpu.d ./Core/Src/lib_mpu.o ./Core/Src/lib_mpu.su ./Core/Src/lib_ov5640.cyclo ./Core/Src/lib_ov5640.d ./Core/Src/lib_ov5640.o ./Core/Src/lib_ov5640.su ./Core/Src/lib_rgb2gray.cyclo ./Core/Src/lib_rgb2gray.d ./Core/Src/lib_rgb2gray.o ./Core/Src/lib_rgb2gray.su ./Core/Src/lib_serial.cyclo ./Core/Src/lib_serial.d ./Core/Src/lib_serial.o ./Core/Src/lib_serial.su ./Core/Src/lib_serialimage.cyclo ./Core/Src/lib_serialimage.d ./Core/Src/lib_serialimage.o ./Core/Src/lib_serialimage.su ./Core/Src/lib_slidingwindow.cyclo ./Core/Src/lib_slidingwindow.d ./Core/Src/lib_slidingwindow.o ./Core/Src/lib_slidingwindow.su ./Core/Src/main.cyclo ./Core/Src/main.d ./Core/Src/main.o ./Core/Src/main.su ./Core/Src/ov5640.cyclo ./Core/Src/ov5640.d ./Core/Src/ov5640.o ./Core/Src/ov5640.su ./Core/Src/ov5640_reg.cyclo ./Core/Src/ov5640_reg.d ./Core/Src/ov5640_reg.o ./Core/Src/ov5640_reg.su ./Core/Src/stm32f7xx_hal_msp.cyclo ./Core/Src/stm32f7xx_hal_msp.d ./Core/Src/stm32f7xx_hal_msp.o ./Core/Src/stm32f7xx_hal_msp.su ./Core/Src/stm32f7xx_it.cyclo ./Core/Src/stm32f7xx_it.d ./Core/Src/stm32f7xx_it.o ./Core/Src/stm32f7xx_it.su ./Core/Src/syscalls.cyclo ./Core/Src/syscalls.d ./Core/Src/syscalls.o ./Core/Src/syscalls.su ./Core/Src/sysmem.cyclo ./Core/Src/sysmem.d ./Core/Src/sysmem.o ./Core/Src/sysmem.su ./Core/Src/system_stm32f7xx.cyclo ./Core/Src/system_stm32f7xx.d ./Core/Src/system_stm32f7xx.o ./Core/Src/system_stm32f7xx.su

.PHONY: clean-Core-2f-Src

