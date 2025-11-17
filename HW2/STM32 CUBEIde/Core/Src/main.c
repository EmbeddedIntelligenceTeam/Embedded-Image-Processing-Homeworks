/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "lib_image.h"
#include "lib_serialimage.h"
#include "string.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
uint32_t g_histogram_data[256];

// İşlenmiş görüntü (filtre sonuçları) için buffer
uint8_t g_processed_image[128 * 128];

// Soru 3b: 3x3 Low Pass Filter (Box Blur) Kerneli
// (Tüm piksellerin ortalamasını alır)
const float g_kernel_low_pass[9] = {
  1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
  1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
  1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f
};

// Soru 3c: 3x3 High Pass Filter (Kenar Bulma) Kerneli
// (Merkez pikseli vurgular, komşuları çıkarır)
const float g_kernel_high_pass[9] = {
  -1.0f, -1.0f, -1.0f,
  -1.0f,  8.0f, -1.0f,
  -1.0f, -1.0f, -1.0f
};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
void Homework_Apply_Histogram_EQ(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);
void Homework_Apply_Convolution(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, const float* kernel);
static void Sort_Bubble_9(uint8_t *arr);
void Homework_Apply_Median_Filter(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
// 128x128 çözünürlük, RGB565 (2 byte) formatı için buffer
volatile uint8_t pImage[128*128];
IMAGE_HandleTypeDef img;
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */
  // Görüntü yapısını 128x128 olarak başlat
  LIB_IMAGE_InitStruct(&img, (uint8_t*)pImage, 128, 128, IMAGE_FORMAT_GRAYSCALE);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

	  //question 1

/*      if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)  // PC -> MCU
	  	{
	  		// 2. (Soru 1) Gelen görüntünün (pImage) histogramını hesapla
	          //    ve sonucu g_histogram_data dizisine yaz
	  		Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, 128, 128);
	  	}
	}    */


	  //question 2

/*      if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)  // PC -> MCU
	  	{
	  		// 2. (Soru 1) Gelen görüntünün ORİJİNAL histogramını hesapla (g_histogram_data'ya yaz)
	  		Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, 128, 128);

	  		// 3. (Soru 2b) Hesaplanan histogramı kullanarak pImage'ı yerinde eşitle
	  		Homework_Apply_Histogram_EQ((uint8_t*)pImage, g_histogram_data, 128, 128);

	  		// 4. (Soru 2c) pImage artık eşitlendi. EŞİTLENMİŞ GÖRÜNTÜNÜN
	          //    yeni histogramını hesapla (g_histogram_data'nın üzerine yaz)
	          Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, 128, 128);

	  		// 5. İşlenmiş (eşitlenmiş) pImage görüntüsünü PC'ye geri gönder
	  	    LIB_SERIAL_IMG_Transmit(&img); // MCU -> PC
	  	}
	}    */


	  //question 3

/*      if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)  // PC -> MCU
	  	{
	  		// 2. (Soru 3b) Gelen görüntüye (pImage) Low Pass Filtre uygula
	          //    Sonucu g_processed_image buffer'ına yaz
	  		Homework_Apply_Convolution(
	              (uint8_t*)pImage,       // Kaynak
	              g_processed_image,      // Hedef
	              128, 128,               // Boyutlar
	              g_kernel_low_pass       // Low Pass Kernel
	          );

	  		// 3. Geri göndermeden önce, işlenmiş görüntüyü ana buffer'a kopyala
	          //    (Çünkü LIB_SERIAL_IMG_Transmit, pImage'ı gönderiyor)
	          memcpy((uint8_t*)pImage, g_processed_image, 128*128);

	  		// 4. İşlenmiş (bulanıklaştırılmış) pImage görüntüsünü PC'ye geri gönder
	  	    LIB_SERIAL_IMG_Transmit(&img); // MCU -> PC
	  	}
	 }     */


      //question 4

/*     if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)  // PC -> MCU
	  	{
	  		// 2. (Soru 4b) Gelen görüntüye (pImage) Medyan Filtre uygula
	          //    Sonucu g_processed_image buffer'ına yaz
	  		Homework_Apply_Median_Filter(
	              (uint8_t*)pImage,       // Kaynak
	              g_processed_image,      // Hedef
	              128, 128                // Boyutlar
	          );

	  		// 3. Geri göndermeden önce, işlenmiş görüntüyü ana buffer'a kopyala
	          memcpy((uint8_t*)pImage, g_processed_image, 128*128);

	  		// 4. İşlenmiş (gürültüsü temizlenmiş) pImage görüntüsünü PC'ye geri gönder
	          //    (Bu, Soru 4c'nin "sonuçları gösterme" [cite: 28] şartını karşılar)
	  	    LIB_SERIAL_IMG_Transmit(&img); // MCU -> PC
	  	}
	}    */

  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 2000000;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height)
{
  uint32_t i;
  uint32_t total_pixels = width * height; // 128 * 128 = 16384

  // 1. Önce histogram dizisinin (g_histogram_data) içini temizle (sıfırla)
  for (i = 0; i < 256; i++)
  {
    p_hist[i] = 0;
  }

  // 2. Görüntünün tüm piksellerini (0'dan 16383'e kadar) tek tek dolaş
  for (i = 0; i < total_pixels; i++)
  {
    // O anki pikselin değerini oku (örn: 150)
    uint8_t pixel_value = p_gray[i];

    // Histogram dizisinde o değere (150'ye) karşılık gelen sayacı 1 arttır
    p_hist[pixel_value]++;
  }
}

void Homework_Apply_Histogram_EQ(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height)
{
  uint32_t i;
  uint32_t total_pixels = width * height; // 16384

  // 1. Kümülatif Dağılım Fonksiyonu (CDF) için bir dizi oluştur
  //    (Bu, 'uint32_t' olmalı çünkü 16384'e kadar çıkabilir)
  static uint32_t cdf[256]; // static olarak tanımlamak stack taşmasını önler
  cdf[0] = p_hist[0];

  for (i = 1; i < 256; i++)
  {
    cdf[i] = cdf[i-1] + p_hist[i];
  }

  // 2. CDF'deki sıfır olmayan ilk (minimum) değeri bul
  uint32_t cdf_min = 0;
  for (i = 0; i < 256; i++)
  {
    if (cdf[i] != 0)
    {
      cdf_min = cdf[i];
      break;
    }
  }

  // 3. Normalizasyon için "Look-Up Table" (LUT) oluştur
  //    Formül: h(v) = round( ( (CDF(v) - CDF_min) * 255 ) / (ToplamPiksel - CDF_min) )

  uint8_t lut[256];

  // Kayan nokta (float) bölme işlemi için ölçek faktörü
  // (ToplamPiksel - cdf_min) 0 olamaz (eğer görüntü boş değilse)
  float scale_factor = 255.0f / (float)(total_pixels - cdf_min);

  for (i = 0; i < 256; i++)
  {
    float h_v = (float)(cdf[i] - cdf_min) * scale_factor;
    // En yakın tam sayıya yuvarla (0.5f eklemek yuvarlama içindir)
    lut[i] = (uint8_t)(h_v + 0.5f);
  }

  // 4. Görüntüyü LUT kullanarak "yerinde" yeniden haritala
  //    (Yani pImage'ın içeriğini kalıcı olarak değiştir)
  for (i = 0; i < total_pixels; i++)
  {
    p_gray[i] = lut[p_gray[i]];
  }
}

void Homework_Apply_Convolution(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, const float* kernel)
{
  uint32_t x, y, kx, ky;
  float sum;
  uint32_t img_idx, kernel_idx;

  // Görüntünün tamamı üzerinde dolaş (y = satır, x = sütun)
  for (y = 0; y < height; y++)
  {
    for (x = 0; x < width; x++)
    {
      // Kenardaki pikselleri (kernelin taşacağı yerleri) atla/sıfırla
      if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
      {
        p_dst[y * width + x] = 0; // Kenarları siyah yap
      }
      else
      {
        sum = 0.0f; // Toplamı sıfırla

        // 3x3'lük Kernel'i dolaş (ky = kernel satır, kx = kernel sütun)
        for (ky = 0; ky < 3; ky++)
        {
          for (kx = 0; kx < 3; kx++)
          {
            // Görüntüdeki ilgili pikselin indeksi
            img_idx = (y + ky - 1) * width + (x + kx - 1);
            // Kernel'deki ilgili ağırlığın indeksi
            kernel_idx = ky * 3 + kx;

            // Toplamı hesapla = (piksel * ağırlık)
            sum += (float)p_src[img_idx] * kernel[kernel_idx];
          }
        }

        // Toplam değeri 0-255 aralığına sıkıştır (Clipping/Clamping)
        if (sum < 0.0f)   sum = 0.0f;
        if (sum > 255.0f) sum = 255.0f;

        // Sonucu hedef buffer'a yaz
        p_dst[y * width + x] = (uint8_t)sum;
      }
    }
  }
}

static void Sort_Bubble_9(uint8_t *arr)
{
    int i, j;
    uint8_t temp;
    // Basit bir bubble sort (9 eleman için yeterince hızlı)
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

/**
  * @brief  Soru 4a: Görüntüye 3x3 Medyan Filtreleme uygular.
  * @param  p_src: Kaynak grayscale görüntü buffer'ı (pImage).
  * @param  p_dst: Hedef grayscale görüntü buffer'ı (g_processed_image).
  * @param  width: Görüntü genişliği
  * @param  height: Görüntü yüksekliği
  */

void Homework_Apply_Median_Filter(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height)
{
  uint32_t x, y, kx, ky;
  uint8_t window[9]; // 3x3'lük pencereyi tutacak dizi
  uint8_t window_idx;

  // Görüntünün tamamı üzerinde dolaş (y = satır, x = sütun)
  for (y = 0; y < height; y++)
  {
    for (x = 0; x < width; x++)
    {
      // Kenardaki pikselleri (pencerenin taşacağı yerleri) atla/sıfırla
      if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
      {
        p_dst[y * width + x] = 0; // Kenarları siyah yap
      }
      else
      {
        window_idx = 0;

        // 1. 3x3'lük penceredeki pikselleri 'window' dizisine kopyala
        for (ky = 0; ky < 3; ky++)
        {
          for (kx = 0; kx < 3; kx++)
          {
            window[window_idx] = p_src[(y + ky - 1) * width + (x + kx - 1)];
            window_idx++;
          }
        }

        // 2. 'window' dizisini sırala
        Sort_Bubble_9(window);

        // 3. Sıralanmış dizinin ortasındaki (medyan) değeri al
        //    (9 eleman için 5. eleman, yani index 4)
        p_dst[y * width + x] = window[4];
      }
    }
  }
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
