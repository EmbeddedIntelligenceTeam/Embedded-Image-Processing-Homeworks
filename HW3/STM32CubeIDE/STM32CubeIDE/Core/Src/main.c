/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body - EE4065 HW3
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
// Görüntü boyutları
#define IMG_WIDTH   128
#define IMG_HEIGHT  128
#define IMG_PIXELS  (IMG_WIDTH * IMG_HEIGHT)


/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
// Histogram verisi
uint32_t g_histogram_data[256];

// İşlenmiş görüntü buffer'ı (convolution / morphology sonuçları)
uint8_t  g_processed_image[IMG_PIXELS];
// Geçici buffer (opening/closing için)
uint8_t  g_tmp_image[IMG_PIXELS];

// Ana görüntü buffer'ı (LIB_SERIALIMAGE bununla çalışıyor)
volatile uint8_t pImage[IMG_PIXELS*2];


IMAGE_HandleTypeDef img;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

// HW2'den gelen fonksiyonlar
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);

// HW3 - Otsu
uint8_t Homework_Compute_Otsu_Threshold(uint32_t* p_hist, uint32_t total_pixels);
void Homework_Apply_Threshold(uint8_t* p_gray, uint32_t width, uint32_t height, uint8_t threshold);

// HW3 - Morphology
void Homework_Dilation_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height);
void Homework_Erosion_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height);
void Homework_Opening_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, uint8_t* p_tmp);
void Homework_Closing_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, uint8_t* p_tmp);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

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

  // 128x128, 8-bit GRAYSCALE görüntü yapısını başlat
  LIB_IMAGE_InitStruct(&img, (uint8_t*)pImage,
                       IMG_HEIGHT, IMG_WIDTH,
                       IMAGE_FORMAT_RGB565);




  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    /************* HW3 Q1 – Otsu Thresholding (Grayscale) **************
     *  - PC'den 128x128 grayscale görüntü gelir.
     *  - Histogram hesaplanır.
     *  - Otsu ile optimal threshold bulunur.
     *  - Görüntü 0 / 255 olacak şekilde binarize edilir.
     *  - Binary görüntü PC'ye geri gönderilir.
     *
     *  NOT: Q2 (colour images) için PC tarafında renkli bir resim
     *  seçip, Python script'i bunu zaten grayscale'e çevirerek
     *  STM32'ye gönderecek. STM32 tarafındaki kod Q1 ile aynıdır.
     *******************************************************************/
#if 0   // Q1'yi çalıştırmak için 1, Q2 ve Q3 testleri için 0 yap.
    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)   // PC -> MCU
    {
      // 1) Histogramı hesapla
      Homework_Calculate_Histogram((uint8_t*)pImage,
                                   g_histogram_data,
                                   IMG_WIDTH, IMG_HEIGHT);

      // 2) Otsu threshold değerini bul
      uint8_t otsu_T =
          Homework_Compute_Otsu_Threshold(g_histogram_data, IMG_PIXELS);

      // 3) Görüntüyü binary hale getir
      Homework_Apply_Threshold((uint8_t*)pImage,
                               IMG_WIDTH, IMG_HEIGHT, otsu_T);

      // 4) Binary görüntüyü PC'ye gönder
      LIB_SERIAL_IMG_Transmit(&img);                // MCU -> PC
    }
#endif

#if 1   // Bu senaryoyu aktifleştirmek için 1 yap, kapatmak için 0 yap
    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)   // PC -> MCU (RGB565 geliyor)
    {
        // 1) pImage içindeki RGB565 veriyi bozmadan, gri kopyasını g_tmp_image'e çıkar

        uint16_t* p_src_rgb = (uint16_t*)pImage;  // 16-bit RGB565 piksel pointer

        for (uint32_t i = 0; i < IMG_PIXELS; i++)
        {
            uint16_t pixel = p_src_rgb[i];

            // RGB565 bitlerini ayıkla
            uint8_t r5 = (pixel >> 11) & 0x1F;
            uint8_t g6 = (pixel >> 5)  & 0x3F;
            uint8_t b5 = (pixel)       & 0x1F;

            // 8-bit'e genişlet (ölçekleme)
            uint8_t r8 = (uint8_t)(r5 << 3);
            uint8_t g8 = (uint8_t)(g6 << 2);
            uint8_t b8 = (uint8_t)(b5 << 3);

            // Basit ortalama ile gri değeri hesapla
            g_tmp_image[i] = (uint8_t)((r8 + g8 + b8) / 3);
        }

        // 2) Histogram & Otsu threshold (g_tmp_image üzerinde)
        Homework_Calculate_Histogram(g_tmp_image,
                                     g_histogram_data,
                                     IMG_WIDTH,
                                     IMG_HEIGHT);

        uint8_t otsu_T =
            Homework_Compute_Otsu_Threshold(g_histogram_data,
                                            IMG_PIXELS);

        // 3) Otsu maskeleme:
        //    gri <= T ise o piksel arka plan: renkli resimde siyah yap
        //    gri  > T ise foreground: rengi olduğu gibi bırak
        for (uint32_t i = 0; i < IMG_PIXELS; i++)
        {
            if (g_tmp_image[i] <= otsu_T)
            {
                // Arka plan: 0x0000 = RGB565 siyah
                p_src_rgb[i] = 0x0000;
            }
            // Aksi durumda p_src_rgb[i] olduğu gibi kalıyor (renkli foreground)
        }

        // 4) Gönderim: pImage zaten RGB565 formatında
        img.format = IMAGE_FORMAT_RGB565;   // Emin olmak için
        LIB_SERIAL_IMG_Transmit(&img);     // MCU -> PC (renkli maske)
    }
#endif
    /************* HW3 Q3 – Morphological Operations (Binary) **********
     *  Aşağıdaki bloklardan her seferinde sadece birini aktif et.
     *  PC'den gelen görüntünün daha önce Otsu ile threshold'lanmış
     *  (0 veya 255 değerli) binary görüntü olduğunu varsayıyoruz.
     *******************************************************************/

#if 0   // Dilation testi için bunu 1 yap, diğerlerini 0 bırak
    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
    {
      Homework_Dilation_3x3((uint8_t*)pImage,
                            g_processed_image,
                            IMG_WIDTH, IMG_HEIGHT);

      memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
      LIB_SERIAL_IMG_Transmit(&img);
    }
#endif

#if 0   // Erosion testi
    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
    {
      Homework_Erosion_3x3((uint8_t*)pImage,
                           g_processed_image,
                           IMG_WIDTH, IMG_HEIGHT);

      memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
      LIB_SERIAL_IMG_Transmit(&img);
    }
#endif

#if 0   // Opening testi (Erosion -> Dilation)
    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
    {
      Homework_Opening_3x3((uint8_t*)pImage,
                           g_processed_image,
                           IMG_WIDTH, IMG_HEIGHT,
                           g_tmp_image);

      memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
      LIB_SERIAL_IMG_Transmit(&img);
    }
#endif

#if 0   // Closing testi (Dilation -> Erosion)
    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
    {
      Homework_Closing_3x3((uint8_t*)pImage,
                           g_processed_image,
                           IMG_WIDTH, IMG_HEIGHT,
                           g_tmp_image);

      memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
      LIB_SERIAL_IMG_Transmit(&img);
    }
#endif

    /* USER CODE END 3 */
  }
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
  huart2.Init.BaudRate   = 2000000;               // 2 Mbps (Python ile uyumlu)
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits   = UART_STOPBITS_1;
  huart2.Init.Parity     = UART_PARITY_NONE;
  huart2.Init.Mode       = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl  = UART_HWCONTROL_NONE;
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
}

/* USER CODE BEGIN 4 */

/* ---------- HW2 Histogram Fonksiyonu (grayscale) ---------- */
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist,
                                  uint32_t width, uint32_t height)
{
  uint32_t i;
  uint32_t total_pixels = width * height;

  for (i = 0; i < 256; i++)
    p_hist[i] = 0;

  for (i = 0; i < total_pixels; i++)
  {
    uint8_t pixel_value = p_gray[i];
    p_hist[pixel_value]++;
  }
}

/* ---------- HW3 – Otsu Threshold Hesabı ---------- */
uint8_t Homework_Compute_Otsu_Threshold(uint32_t* p_hist, uint32_t total_pixels)
{
  uint32_t sum_all = 0;
  uint32_t sumB = 0;
  uint32_t wB = 0;
  uint32_t wF = 0;
  float mB, mF;
  float var_between;
  float max_var_between = 0.0f;
  uint8_t threshold = 0;

  // 1) Σ i * h[i]
  for (uint32_t i = 0; i < 256; i++)
    sum_all += i * p_hist[i];

  // 2) Tüm olası eşikler için sınıflar arası varyans
  for (uint32_t t = 0; t < 256; t++)
  {
    wB += p_hist[t];
    if (wB == 0)
      continue;

    wF = total_pixels - wB;
    if (wF == 0)
      break;

    sumB += t * p_hist[t];

    mB = (float)sumB / (float)wB;
    mF = (float)(sum_all - sumB) / (float)wF;

    float diff = mB - mF;
    var_between = (float)wB * (float)wF * diff * diff;

    if (var_between > max_var_between)
    {
      max_var_between = var_between;
      threshold = (uint8_t)t;
    }
  }

  return threshold;
}

/* ---------- HW3 – Uygulama: Threshold'a göre binarize et ---------- */
void Homework_Apply_Threshold(uint8_t* p_gray, uint32_t width,
                              uint32_t height, uint8_t threshold)
{
  uint32_t total_pixels = width * height;

  for (uint32_t i = 0; i < total_pixels; i++)
  {
    uint8_t val = p_gray[i];
    if (val > threshold)
      p_gray[i] = 255;
    else
      p_gray[i] = 0;
  }
}

/* ---------- HW3 – Morphology: Dilation ---------- */
void Homework_Dilation_3x3(uint8_t* p_src, uint8_t* p_dst,
                           uint32_t width, uint32_t height)
{
  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
      {
        p_dst[y * width + x] = 0;
      }
      else
      {
        uint8_t max_val = 0;

        for (int ky = -1; ky <= 1; ky++)
        {
          for (int kx = -1; kx <= 1; kx++)
          {
            uint8_t v = p_src[(y + ky) * width + (x + kx)];
            if (v > max_val)
              max_val = v;
          }
        }

        p_dst[y * width + x] = max_val;
      }
    }
  }
}

/* ---------- HW3 – Morphology: Erosion ---------- */
void Homework_Erosion_3x3(uint8_t* p_src, uint8_t* p_dst,
                          uint32_t width, uint32_t height)
{
  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
      {
        p_dst[y * width + x] = 0;
      }
      else
      {
        uint8_t min_val = 255;

        for (int ky = -1; ky <= 1; ky++)
        {
          for (int kx = -1; kx <= 1; kx++)
          {
            uint8_t v = p_src[(y + ky) * width + (x + kx)];
            if (v < min_val)
              min_val = v;
          }
        }

        p_dst[y * width + x] = min_val;
      }
    }
  }
}

/* ---------- HW3 – Morphology: Opening (Erosion -> Dilation) ---------- */
void Homework_Opening_3x3(uint8_t* p_src, uint8_t* p_dst,
                          uint32_t width, uint32_t height,
                          uint8_t* p_tmp)
{
  Homework_Erosion_3x3(p_src, p_tmp, width, height);
  Homework_Dilation_3x3(p_tmp, p_dst, width, height);
}

/* ---------- HW3 – Morphology: Closing (Dilation -> Erosion) ---------- */
void Homework_Closing_3x3(uint8_t* p_src, uint8_t* p_dst,
                          uint32_t width, uint32_t height,
                          uint8_t* p_tmp)
{
  Homework_Dilation_3x3(p_src, p_tmp, width, height);
  Homework_Erosion_3x3(p_tmp, p_dst, width, height);
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
  /* place for user assert code */
}
#endif /* USE_FULL_ASSERT */
