C:::::::::::::::::::::::::::::: HEUVAC ::::::::::::::::::::::::::::::::::::::
C.. High resolution version of the EUVAC model. 
C.. This routine returns 1) the solar EUV fluxes and 2) the flux
C.. weighted photoionization cross sections
C.. The EUVAC flux model uses the F74113 solar reference spectrum and 
C.. ratios determined from the AE-E satellite. It uses the daily
C.. F10.7 flux (F107) and the 81 day mean (F107A) as a proxy for scaling
C.. the fluxes. This version sums the fluxes into user specified bins
C.. The solar activity scaling uses the same 37 scaling factors as the EUVAC
C.. model. See Richards et al. [1994] J. Geophys. Res. p8981 for EUVAC details.
C.. Written by Phil Richards, March 2004

      SUBROUTINE HEUVAC(IDIM,   !.. in: Array dimensions
     >                  F107,   !.. in: daily 10.7 cm flux index. 
     >                 F107A,   !.. in: 81 day average of daily F10.7 
     >                  KMAX,   !.. in: number of bins for the flux
     >                BINLAM,   !.. in: the wavelength (Angstrom) bin boundaries 
     >              BIN_ANGS,   !.. out: fluxes (photons/cm2/s) in the bins
     >                XS_OUT,   !.. out: Weighted cross section in the bins
     >               TORR_WL,   !.. out: the wavelengths in 37 Torr bins
     >              TORR_FLX)   !.. out: the fluxes in 37 Torr bins

      IMPLICIT NONE
	INTEGER IDIM                     !.. array dimensions
	INTEGER I,K                      !.. loop control variables
      INTEGER IFLUX                    !.. number of fluxes in reference spectrum
      INTEGER FREAD                    !.. signifies read status of reference spectrum
      INTEGER KMAX                     !.. maximum bins and reference factor switch
	REAL FLXFAC                      !.. solar activity factor
      REAL F107,F107A                  !.. solar activity indices
      REAL F10SAVE                     !.. previous solar activity index
      REAL LAM_EUV(IDIM),FLUX(IDIM)    !.. wavelength and scaled flux
      REAL FLUX_REF(IDIM)              !.. Reference flux, no F10.7 scaling
	INTEGER IWLXS                    !.. Number of photoionization cross sections
	REAL WLXS(IDIM),XS(15,IDIM)      !.. Photoionization wavelengths and cross sections 
      REAL BINLAM(IDIM),BIN_ANGS(IDIM) !.. Bins for summing fluxes
      REAL XS_OUT(15,IDIM)             !.. Weighted cross section
      REAL TORR_WL(37),TORR_FLX(37)    !.. Wavelengths and EUV flux in Torr 37 bins

	!.. If F107 hasn't changed, return
	DATA F10SAVE/-9/
	IF(F107+F107A.EQ.F10SAVE) RETURN
      F10SAVE=F107+F107A

      !.. On the first call only, get reference F74113 spectrum and cross sections
	DATA FREAD/-9/
	IF(FREAD.EQ.-9) THEN
	  FREAD=1
        CALL GET_F74113(IDIM,IFLUX,LAM_EUV,FLUX_REF)
        !.. Apply factors below 250 A 
        CALL MOD_F74113(IDIM,IFLUX,LAM_EUV,FLUX_REF)

	  !.. Get the Fennelly and Torr photoionization cross sections
	  CALL GET_PHOTO_XS(IWLXS,WLXS,XS)
	ENDIF

      !.. Apply multiplicative solar activity factors to fluxes
	DO I=1,IFLUX
	  FLXFAC=1.0
        CALL CAL_EUV_FAC(LAM_EUV(I),F107,F107A,FLXFAC)
        FLUX(I)=FLUX_REF(I)*FLXFAC
      ENDDO

      !.. Bin the fluxes into user specified wavelength bins
      CALL BIN_FLUX(IDIM,IFLUX,KMAX,LAM_EUV,FLUX,BINLAM,BIN_ANGS)

	!.. calculate the flux weighted cross sections
      CALL BIN_XS_WL(IDIM,IWLXS,WLXS,XS,IFLUX,LAM_EUV,FLUX,
     >  KMAX,BINLAM,XS_OUT)

      !... put the fluxes in the Torr wavelength bins
      CALL BIN_TORR(13,IDIM,IFLUX,LAM_EUV,FLUX,TORR_WL,TORR_FLX)

      RETURN
	END
C:::::::::::::::::::::::: BIN_XS_WL:::::::::::::::::::::::::::::::::::
C.. This routine calculates flux weighted cross sections into user
C.. specified bins  
C.. Written by Phil Richards, March 2004
      SUBROUTINE BIN_XS_WL(IDIM, !.. array dimensions
     >                    IWLXS, !.. number of cross sections
     >                     WLXS, !.. XS wavelengths
     >                       XS, !.. cross sections
     >                    IFLUX, !.. number of reference fluxes
     >                  LAM_EUV, !.. the reference wavelengths
     >                     FLUX, !.. the reference fluxes 
     >                     KMAX, !.. number of bins for the flux
     >                   BINLAM, !.. the wavelength (Angstrom) bin boundaries 
     >                   XS_OUT) !.. fluxes (photons/cm2/s) in the bins

      IMPLICIT NONE
	INTEGER I,IDIM,K,IWLXS,IFLUX,KW,J,KMAX
	REAL WLXS(IDIM),XS(15,IDIM)
	REAL LAM_EUV(IDIM),FLUX(IDIM),XS_OUT(15,IDIM),EV,TFLUX
      REAL SUM_FLUX(IDIM)
	REAL BINLAM(KMAX)

	DO K=1,KMAX
	  SUM_FLUX(K)=0.0
	  DO I=1,15
          XS_OUT(I,K)=0.0
	  ENDDO
	ENDDO

      I=0 
      K=1     
      !.. Sum the fluxes into 1 eV bins
 20	I=I+1
        IF(I.GT.IFLUX) GO TO 30
	  IF(LAM_EUV(I).GT.BINLAM(K+1)) K=K+1
        IF(K.GT.KMAX-1) GO TO 30
          !.. Find the wavelength index in the cross section array
          CALL BISPLT(1,IWLXS,IDIM,LAM_EUV(I),WLXS,KW)
	    TFLUX=FLUX(I)+1.000   !.. 1 is added in case no flux in bin
	    SUM_FLUX(K)=SUM_FLUX(K)+TFLUX
	    DO J=1,15
	      XS_OUT(J,K)=XS_OUT(J,K)+TFLUX*XS(J,KW)
          ENDDO
	GO TO 20
 30   CONTINUE

      !.. Normalize the cross sections
	DO K=1,KMAX-1
	  IF(SUM_FLUX(K).GE.10) THEN
	    DO J=1,15
           XS_OUT(J,K)=XS_OUT(J,K)/SUM_FLUX(K)
	    ENDDO
	  ENDIF
      ENDDO

      RETURN
	END
C:::::::::::::::::::::::::::::::: CAL_EUV_FAC :::::::::::::::::::::::
C...... Calculate the solar activity factors
C...... This routine uses the ratios determined from Hinteregger's 
C...... SERF1 model to scale the solar flux using the daily
C...... F10.7 flux (F107) and the 81 day mean (F107A) as a proxy.
C...... The factors are specified in the he 37 wavelength
C...... bins of Torr et al. [1979] Geophys. Res. Lett. p771. 
C...... See Richards et al. [1994] J. Geophys. Res. p8981 for details.
C...... Written by Phil Richards, March 2004
      SUBROUTINE CAL_EUV_FAC(LAMBDA,   !.. in: wavelength to scale
     >                         F107,   !.. in: daily solar activity
     >                        F107A,   !.. in: Average solar activity
     >                       FLXFAC)   !.. out: flux multiplier
      IMPLICIT NONE
      INTEGER I,NLINES,NCELLS  !.. loop control, # of cells and lines
      PARAMETER (NLINES=19,NCELLS=22) 
      REAL F107,F107A       !.. in: daily solar activity
      REAL FLXFAC,LAMBDA    !.. see definitions above
      REAL ZCELLS(NCELLS)   !.. wavelengths for the 50 A cells
      REAL ACELLS(NCELLS)   !.. A coeffs for the 50 A cells
      REAL ZLINES(NLINES)   !.. wavelengths for the lines
      REAL ALINES(NLINES)   !.. A coeffs for the lines

      !.. Boundary wavelengths for the 50 A bins
      DATA ZCELLS/0,50,100,150,200,250,300,350,400,450,500,550,600,
     >  650,700,750,800,850,900,950,1000,1050/
      !.. A coeffs for the 50 A bins. Note that the 0-50 A scaling factor
	!.. is from Hinteregger's model
      DATA ACELLS/.05,1.0017E-02,7.1250E-03,1.3375E-02,
     > 1.9450E-02,2.6467E-02,2.5000E-03,3.6542E-02,7.4083E-03,
     > 2.0225E-02,8.7583E-03,3.6583E-03,1.1800E-02,4.2667E-03,
     > 4.7500E-03,4.7667E-03,4.8167E-03,5.6750E-03,4.9833E-03,
     > 4.4167E-03,4.3750E-03,4.3750E-03/
      !.. Wavelengths of the discrete lines. 335.41 A line added with
	!.. same scaling factor as 284.15 A line (both Fe lines). Note that
	!.. the ACELL factor for the 300-350 bin had to be reduced to 2.5000E-03
	!.. to preserve the total flux in that bin.
      DATA ZLINES/256.3,284.15,303.31,303.78,335.41,368.07,465.22,
     >  554.37,584.33,609.76,629.73,703.36,765.15,770.41,787.71,
     >  790.15,977.02,1025.72,1031.91/
      !.. A coefficients for the discrete lines
      DATA ALINES/2.78E-03,1.38E-01,2.50E-02,3.33E-03,1.38E-01,6.59E-03,
     >  7.49E-03,3.27E-03,5.16E-03,1.62E-02,3.33E-03,3.04E-03,
     >  3.85E-03,1.28E-02,3.28E-03,3.28E-03,3.94E-03,5.18E-03,
     >  5.28E-03/

      !.. use binary splitting to find the location of LAMBDA in
      !.. the lines array
      CALL BISPLT(1,NLINES,NLINES,LAMBDA,ZLINES,I)

	!.. check to see if this LAMBDA is one of the lines
      IF(ABS(LAMBDA-ZLINES(I)).LT.0.05) THEN
        FLXFAC=(1.0 + ALINES(I) * (0.5*(F107+F107A) - 80.0))
	ELSE
        !.. not a line so use binary splitting to find the correct 50 A bin
        CALL BISPLT(1,NCELLS,NCELLS,LAMBDA,ZCELLS,I)
        FLXFAC=(1.0 + ACELLS(I) * (0.5*(F107+F107A) - 80.0))
	ENDIF

      IF(FLXFAC.LT.0.8) FLXFAC=0.8  !.. don't let FLXFAC get too small

      RETURN
      END

C::::::::::::::::::::::::::: BISPLT ::::::::::::::::::::::::::::::::::
C--- Use bisection to find the nearest altitude ZARRAY(K) to desired
C--- altitude (Z). Modified in Feb 93 to do both ascending and descending
C--- arrays. Written by Phil Richards, modified March 2004
      SUBROUTINE BISPLT(BEGIN,  !.. first index in array
     >                  FINAL,  !.. ending index in array
     >                   IDIM,  !.. array dimension
     >                      Z,  !.. scalar to look for
     >                 ZARRAY,  !.. array to search
     >                  FIRST)  !.. index of Z in ZARRAY
      IMPLICIT NONE
      INTEGER BEGIN,IDIM,FINAL
      INTEGER FIRST,LAST,MID !.. variable indices of sub arrays
      REAL ZARRAY(IDIM),Z

      FIRST=BEGIN  !.. set pointer to first element
      LAST=FINAL   !.. set pointer to last element

      !.. This section for ZARRAY increasing
      IF(ZARRAY(BEGIN).LE.ZARRAY(FINAL)) THEN
 10      CONTINUE
             MID=(FIRST+LAST)/2
             IF(Z.LT.ZARRAY(MID)) THEN
                LAST=MID-1
             ELSE
                FIRST=MID+1
             ENDIF  
         IF(FIRST.LT.LAST) GO TO 10
         IF(Z.LT.ZARRAY(FIRST)) FIRST=FIRST-1
         IF(Z.LT.ZARRAY(BEGIN)) FIRST=BEGIN

      !.. This section for ZARRAY decreasing
      ELSE
 110      CONTINUE
             MID=(FIRST+LAST)/2
             IF(Z.GE.ZARRAY(MID)) THEN
                LAST=MID-1
             ELSE
                FIRST=MID+1
             ENDIF  
         IF(FIRST.LT.LAST) GO TO 110
         IF(Z.GT.ZARRAY(FIRST)) FIRST=FIRST-1
         IF(Z.LT.ZARRAY(FINAL)) FIRST=FINAL
      ENDIF

      RETURN
      END
 
C:::::::::::::::::::::: BIN_FLUX ::::::::::::::::::::
C...... This routine puts the fluxes into user specified bins
C...... Written by Phil Richards, March 2004
      SUBROUTINE BIN_FLUX(IDIM,   !.. in: array dimensions
     >                   IFLUX,   !.. in: number of reference fluxes
     >                    KMAX,   !.. in: number of bins for the flux
     >                 LAM_EUV,   !.. in: the reference wavelengths
     >                    FLUX,   !.. in: the reference fluxes 
     >                  BINLAM,   !.. in: the wavelength (Angstrom) bin boundaries 
     >                  BIN_ANGS) !.. out: fluxes (photons/cm2/s) in the bins

      IMPLICIT NONE
	INTEGER IDIM,IFLUX,I,K,KMAX
	REAL LAM_EUV(IDIM),FLUX(IDIM),BIN_ANGS(IDIM),BINLAM(IDIM)
      !.. Sum the fluxes into specified bins
      I=1
       
	!..small background flux for plotting
      DO K=1,KMAX
        BIN_ANGS(K)=1.0E1
        IF(BINLAM(K).GT.LAM_EUV(IFLUX)) GO TO 5  
	ENDDO

 5    CONTINUE
      !.. readjust KMAX so max wavelength <= maximum wavelength in input spectrum 
      KMAX=K-1

	!.. loop through bins summing fluxes
	DO K=1,KMAX
 10     CONTINUE
	  IF(I.LE.IFLUX) THEN
 	    IF(LAM_EUV(I).GT.BINLAM(K).AND.LAM_EUV(I).LE.BINLAM(K+1)) THEN
	      BIN_ANGS(K)=BIN_ANGS(K)+FLUX(I)
	      I=I+1
	      GOTO 10
	    ENDIF
	  ENDIF
      ENDDO
      RETURN
	END
C:::::::::::::::::::::::: MOD_F74113 :::::::::::::::::::::::::::::::::::
C...... This routine applies the EUVAC correction below 250 A. 
C...... Some of the fluxes in the 50 eV bins have been normalized to bring 
C...... them into agreement with the 37 bin F74113 spectrum published 
C...... by Torr et al. [1979] GRL page 771
C...... Written by Phil Richards, March 2004
      SUBROUTINE MOD_F74113(IDIM, !.. Array dimension
     >                     IFLUX, !.. number of fluxes to return
     >                   LAM_EUV, !.. F74113 wavelengths
     >                      FLUX) !.. F74113 fluxes
      IMPLICIT NONE
	INTEGER I,IDIM,IFLUX
	REAL LAM_EUV(IDIM),FLUX(IDIM)
      
	!.. Adjust fluxes to EUVAC model values
      DO I=1,IFLUX
	  !.. Adjust fluxes  in some ranges to get Torr et al. [1979] 
        !.. F74113 spectrum fluxes
        IF(LAM_EUV(I).GT.500.AND.LAM_EUV(I).LE.550.AND.
     >    FLUX(I).LT.1.0E8) FLUX(I)=FLUX(I)*1.18
        IF(LAM_EUV(I).GT.550.AND.LAM_EUV(I).LE.600.AND.
     >    FLUX(I).LT.1.0E8) FLUX(I)=FLUX(I)*1.9
        IF(LAM_EUV(I).GT.600.AND.LAM_EUV(I).LE.650.AND.
     >    FLUX(I).LT.1.0E8) FLUX(I)=FLUX(I)*1.65
        IF(LAM_EUV(I).GT.650.AND.LAM_EUV(I).LE.700.AND.
     >    FLUX(I).LT.1.0E8) FLUX(I)=FLUX(I)*1.7
        IF(LAM_EUV(I).GT.700.AND.LAM_EUV(I).LE.750.AND.
     >    FLUX(I).LT.1.0E8) FLUX(I)=FLUX(I)*1.1
        IF(LAM_EUV(I).GT.750.AND.LAM_EUV(I).LE.800.AND.
     >    FLUX(I).LT.1.0E8) FLUX(I)=FLUX(I)*1.12
        IF(LAM_EUV(I).GT.800.AND.LAM_EUV(I).LE.850.AND.
     >    FLUX(I).LT.1.0E8) FLUX(I)=FLUX(I)*1.01

        !.. multiply fluxes below 250 A for EUVAC model
        IF(LAM_EUV(I).GT.150.AND.LAM_EUV(I).LE.250)
     >     FLUX(I)=FLUX(I)*2.0
        IF(LAM_EUV(I).GT.25.AND.LAM_EUV(I).LE.150)
     >     FLUX(I)=FLUX(I)*3.0
        IF(LAM_EUV(I).GT.0.0.AND.LAM_EUV(I).LE.25)
     >     FLUX(I)=FLUX(I)*9.0
	ENDDO
	RETURN
	END


C:::::::::::::::::::::::: BIN_TORR:::::::::::::::::::::::::::::::::::
C....... This routine sums the solar fluxes into the Torr et al. 37 wavelength
C....... bins for EUVAC
C....... Torr, M. R., D. G. Torr, R. A. Ong, and H. E. Hinteregger, Ionization 
C....... frequencies for major thermospheric constituents as a function of solar 
C....... cycle 21, Geophys. Res. Lett., 6, 771, 1979.
C...... Written by Phil Richards, March 2004

      SUBROUTINE BIN_TORR(IFILE, !.. in: File to write
     >                     IDIM, !.. in: array dimensions
     >                    IFLUX, !.. in: number of reference fluxes
     >                  LAM_EUV, !.. in: the reference wavelengths
     >                     FLUX, !.. in: the reference fluxes 
     >                  TORR_WL, !.. out: the wavelengths in 37 Torr bins
     >                 TORR_FLX) !.. out: the fluxes in 37 Torr bins
      IMPLICIT NONE
      INTEGER I,IDIM,LMAX,K,IFLUX,L,IFILE
	PARAMETER (LMAX=37)
      INTEGER ZBIN(LMAX),KNO(LMAX)
	REAL LAM_EUV(IDIM),FLUX(IDIM),ZLX(LMAX),TORR_FLX(LMAX),LINE(IDIM),
     >   LAM_LO(LMAX),LAM_HI(LMAX),LAM_DEL(LMAX),F74113(LMAX),TLAM,FSUM
	REAL TORR_WL(LMAX)  !.. for transferring back to calling program


      !.. The Torr et al. F74113 spectrum fluxes (1.0E-9)
      DATA F74113/2.467,2.1,3.5,1.475,4.4,3,3.537,1.625,0.758,0.702,
     >  0.26,0.17,0.141,0.36,0.23,0.342,1.59,0.53,0.357,1.27,0.72,
     >  0.452,0.285,0.29,0.383,0.314,0.65,0.965,6.9,0.8,1.679,0.21,
     >  0.46,3.1,4.8,0.45,1.2/
          
      !.. The Torr et al. F74113 spectrum bins
      DATA ZLX/1025.,1031.91,1025.72,975.,977.02,925.,875.,825.,775.,
     > 789.36,770.41,765.15,725.,703.36,675.,625.,629.73,609.76,575.,
     > 584.33,554.31,525.,475.,465.22,425.,375.,368.07,325.,303.78,
     > 303.31,275.,284.15,256.32,225.,175.,125.,75./

      !.. Bin widths used to pick off the lines
      DATA LAM_DEL/25.,.2,.2,25.,.2,25.,25.,25.,25.,2.3,.2,.2,25.,
     > .2,25.,25.,.2,.2,25.,.2,.2,25.,25.,.2,25.,25.,.2,25.,.2,
     > .2,25.,.2,.2,25.,25.,25.,25./

      !.. Upper wavelength of the wavelength bins
      DATA LAM_HI/1050.,1034.91,1028.72,1000.,980.02,950.,900.,850.,
     > 800.,792.36,773.41,768.15,750.,706.36,700.,650.,632.73,612.76,
     > 600.,587.33,557.31,550.,500.,468.22,450.,400.,371.07,350.,306.78,
     > 303.77,300.,287.15,259.3,250.,200.,150.,100./

      !.. Lower wavelength of the wavelength bins
      DATA LAM_LO/1000.,1028.91,1022.72,950.,974.02,900.,850.,800.,
     > 750.,786.36,767.41,762.15,700.,700.36,650.,600.,626.73,606.76,
     > 550.,581.33,551.31,500.,450.,462.22,400.,350.,365.07,300.,303.32,
     > 298.31,250.,281.15,253.3,200.,150.,100.,50./

      !.. marks bins (1) and lines (0)
      DATA ZBIN/1,0,0,1,0,1,1,1,1,0,0,0,1,0,1,1,0,0,1,0,0,1,1,0,1,
     >   1,0,1,0,0,1,0,0,1,1,1,1/
	DATA KNO/LMAX*1/  !.. used to prevent adding fluxes to the bins for lines

      DO I=1,IFLUX
	  LINE(I)=1.0
	ENDDO

	DO K=1,LMAX
        TORR_FLX(K)=0.0
	  TORR_WL(K)=ZLX(K) 
	ENDDO

      !... Put the fluxes for the lines into their Wavelength bins and
      !.. mark them out so they are not counted in the 50 A wavelength bins    
	DO K=1,LMAX
	  !.. If this bin is a line look for fluxes to add
	  IF(ZBIN(K).EQ.0) THEN
	    DO I=1,IFLUX
            TLAM=LAM_EUV(I)
	      !.. This part for F74113
	      IF(ABS(TLAM-ZLX(K)).LT.LAM_DEL(K)) THEN 
              TORR_FLX(K)=TORR_FLX(K)+FLUX(I)*LINE(I)
	        LINE(I)=0.0          !.. zero so lines are not added twice
              KNO(K)=0
	      ENDIF
          ENDDO
	  ENDIF
      ENDDO

      !... Put the remaining fluxes in their 50 A Wavelength bins 
	DO I=1,IFLUX
	  DO K=1,LMAX
          IF(LINE(I).GT.0.1.AND.KNO(K).GT.0) THEN
            IF(LAM_EUV(I).LT.LAM_HI(K).AND.LAM_EUV(I).GE.LAM_LO(K)) THEN
               TORR_FLX(K)=TORR_FLX(K)+FLUX(I)*LINE(I)
	         LINE(I)=0.0          !.. zero so lines are not added twice
	      ENDIF
	    ENDIF
        ENDDO
      ENDDO
      
	FSUM=0
	IF(FSUM.NE.-1) RETURN !.. aborts write statements

  	WRITE(IFILE,90)       
 90   FORMAT(3X,'  Fluxes in 37 Torr and Torr bins'
     >  ,/3X,'K     ZLX     LAMHI    LAMLO  LAMHI_LAMLO SUM37'
     >   ,1X,'  F74113    RATIO    DEL_F    Flux(ph/cm2/s)')

      DO K=1,LMAX
	 L=LMAX+1
  	 WRITE(IFILE,'(I5,8F9.2,1P,9E13.3)') K,ZLX(L-K),LAM_HI(L-K),
     >   LAM_LO(L-K),LAM_HI(L-K)-LAM_LO(L-K),
     >   TORR_FLX(L-K)/1.0E9,F74113(L-K),
     >   TORR_FLX(L-K)/F74113(L-K)/1.0E9,
     >   TORR_FLX(L-K)/1.0E9-F74113(L-K),TORR_FLX(L-K)
         FSUM=FSUM+TORR_FLX(L-K)/1.0E9-F74113(L-K)
	ENDDO

      RETURN
	END

C:::::::::::::::::::::::: GET_F74113 :::::::::::::::::::::::::::::::::::
C...... This routine returns the Hinteregger F74113 reference solar flux.
C...... Written by Phil Richards, March 2004
      SUBROUTINE GET_F74113(IDIM, !.. Array dimension
     >                     IFLUX, !.. number of fluxes to return
     >                   LAM_EUV, !.. F74113 wavelengths
     >                      FLUX) !.. F74113 fluxes
      IMPLICIT NONE
	INTEGER I,IDIM,IFLUX,NFLUX
	PARAMETER(NFLUX=969)           !.. number of fluxes in data statements
	REAL LAM_EUV(IDIM),FLUX(IDIM),
     >  L74113(NFLUX),F74113(NFLUX)  !.. arrays for the F74113 data

      IFLUX=NFLUX       !.. copy NFLUX to IFLUX to return
	!.. Transfer fluxes to return arrays
      DO I=1,IFLUX
        LAM_EUV(I)=L74113(I)
        FLUX(I)=F74113(I)
	ENDDO
	RETURN

      DATA L74113/      !.. Wavelengths
     > 14.25,14.40,15.01,15.26,16.01,16.77,17.05,17.11,
     > 18.62,18.97,21.60,21.80,22.10,28.47,28.79,29.52,
     > 30.02,30.43,33.74,40.95,43.76,44.02,44.16,45.66,
     > 46.40,46.67,47.87,49.22,50.36,50.52,50.69,52.30,
     > 52.91,54.15,54.42,54.70,55.06,55.34,56.08,56.92,
     > 57.36,57.56,57.88,58.96,59.62,60.30,60.85,61.07,
     > 61.63,61.90,62.30,62.35,62.77,62.92,63.16,63.30,
     > 63.65,63.72,64.11,64.60,65.21,65.71,65.85,66.30,
     > 67.14,67.35,68.35,69.65,70.0,70.54,70.75,71.0,
     > 71.94,72.31,72.63,72.80,72.95,73.55,74.21,74.44,
     > 74.83,75.03,75.29,75.46,75.73,76.01,76.48,76.83,
     > 76.94,77.30,77.74,78.56,78.70,79.08,79.48,79.76,
     > 80.0,80.21,80.55,80.94,81.16,81.58,81.94,82.43,
     > 82.67,83.25,83.42,83.67,84.0,84.26,84.50,84.72,
     > 84.86,85.16,85.50,85.69,85.87,86.23,86.40,86.77,
     > 86.98,87.30,87.61,88.10,88.11,88.14,88.42,88.64,
     > 88.90,89.14,89.70,90.14,90.45,90.71,91.0,91.48,
     > 91.69,91.81,92.09,92.55,92.81,93.61,94.07,94.25,
     > 94.39,94.81,94.90,95.37,95.51,95.81,96.05,96.49,
     > 96.83,97.12,97.51,97.87,98.12,98.23,98.50,98.88,
     > 99.44,99.71,99.99,100.54,100.96,101.57,102.15,103.01,
     > 103.15,103.17,103.58,103.94,104.23,104.76,105.23,106.25,
     > 106.57,106.93,108.05,108.46,109.50,109.98,110.56,110.62,
     > 110.76,111.16,111.25,113.80,114.09,114.24,115.39,115.82,
     > 116.75,117.20,120.40,121.15,121.79,122.70,123.50,127.65,
     > 129.87,130.30,131.02,131.21,136.21,136.28,136.34,136.45,
     > 136.48,141.20,144.27,145.04,148.40,150.10,152.15,154.18,
     > 157.73,158.37,159.98,160.37,164.15,167.50,168.17,168.55,
     > 168.92,169.70,171.08,172.17,173.08,174.58,175.26,177.24,
     > 178.05,179.27,179.75,180.41,181.14,182.17,183.45,184.53,
     > 184.80,185.21,186.60,186.87,187.95,188.23,188.31,190.02,
     > 191.04,191.34,192.40,192.82,193.52,195.13,196.52,196.65,
     > 197.44,198.58,200.02,201.13,202.05,202.64,203.81,204.25,
     > 204.94,206.26,206.38,207.46,208.33,209.63,209.78,211.32,
     > 212.14,213.78,214.75,215.16,216.88,217.0,218.19,219.13,
     > 220.08,221.44,221.82,224.74,225.12,227.01,227.19,227.47,
     > 228.70,230.65,231.55,232.60,233.84,234.38,237.33,239.87,
     > 240.71,241.74,243.03,243.86,244.92,245.94,246.24,246.91,
     > 247.18,249.18,251.10,251.95,252.19,253.78,256.32,256.38,
     > 256.64,257.16,257.39,258.36,259.52,261.05,262.99,264.24,
     > 264.80,270.51,271.99,272.64,274.19,275.35,275.67,276.15,
     > 276.84,277.0,277.27,278.40,281.41,284.15,285.70,289.17,
     > 290.69,291.70,292.78,296.19,299.50,303.31,303.78,315.02,
     > 316.20,319.01,319.83,320.56,335.41,345.13,345.74,347.39,
     > 349.85,356.01,360.80,364.48,368.07,399.82,401.14,401.94,
     > 403.26,405.0,406.0,407.0,408.0,409.0,410.0,411.0,
     > 412.0,413.0,414.0,415.0,416.0,417.0,417.24,418.0,
     > 419.0,420.0,421.0,422.0,423.0,424.0,425.0,426.0,
     > 427.0,428.0,429.0,430.0,430.47,431.0,432.0,433.0,
     > 434.0,435.0,436.0,436.70,437.0,438.0,439.0,440.0,
     > 441.0,442.0,443.0,444.0,445.0,446.0,447.0,448.0,
     > 449.0,450.0,451.0,452.0,453.0,454.0,455.0,456.0,
     > 457.0,458.0,459.0,460.0,461.0,462.0,463.0,464.0,
     > 465.0,465.22,466.0,467.0,468.0,469.0,470.0,471.0,
     > 472.0,473.0,474.0,475.0,476.0,477.0,478.0,479.0,
     > 480.0,481.0,482.0,483.0,484.0,485.0,486.0,487.0,
     > 488.0,489.0,489.50,490.0,491.0,492.0,493.0,494.0,
     > 495.0,496.0,497.0,498.0,499.0,499.37,500.0,501.0,
     > 502.0,503.0,504.0,507.93,515.60,520.66,525.80,537.02,
     > 554.37,558.60,562.80,584.33,599.60,608.0,609.0,609.76,
     > 610.0,611.0,612.0,613.0,614.0,615.0,616.0,616.60,
     > 617.0,618.0,619.0,620.0,621.0,622.0,623.0,624.0,
     > 624.93,625.0,626.0,627.0,628.0,629.0,629.73,630.0,
     > 631.0,632.0,633.0,634.0,635.0,636.0,637.0,638.0,
     > 639.0,640.0,640.41,640.93,641.0,641.81,642.0,643.0,
     > 644.0,645.0,646.0,647.0,648.0,649.0,650.0,651.0,
     > 652.0,653.0,654.0,655.0,656.0,657.0,657.30,658.0,
     > 659.0,660.0,661.0,661.40,662.0,663.0,664.0,665.0,
     > 666.0,667.0,668.0,669.0,670.0,671.0,672.0,673.0,
     > 674.0,675.0,676.0,677.0,678.0,679.0,680.0,681.0,
     > 682.0,683.0,684.0,685.0,685.71,686.0,687.0,688.0,
     > 689.0,690.0,691.0,692.0,693.0,694.0,695.0,696.0,
     > 697.0,698.0,699.0,700.0,701.0,702.0,703.0,703.36,
     > 704.0,705.0,706.0,707.0,708.0,709.0,710.0,711.0,
     > 712.0,713.0,714.0,715.0,716.0,717.0,718.0,718.50,
     > 719.0,720.0,721.0,722.0,723.0,724.0,725.0,726.0,
     > 727.0,728.0,729.0,730.0,731.0,732.0,733.0,734.0,
     > 735.0,736.0,737.0,738.0,739.0,740.0,741.0,742.0,
     > 743.0,744.0,745.0,746.0,747.0,748.0,749.0,750.0,
     > 751.0,752.0,753.0,754.0,755.0,756.0,757.0,758.0,
     > 758.68,759.0,759.44,760.0,760.30,761.0,761.13,762.0,
     > 762.0,763.0,764.0,765.0,765.15,766.0,767.0,768.0,
     > 769.0,770.0,770.41,771.0,772.0,773.0,774.0,775.0,
     > 776.0,776.0,777.0,778.0,779.0,780.0,780.32,781.0,
     > 782.0,783.0,784.0,785.0,786.0,786.47,787.0,787.71,
     > 788.0,789.0,790.0,790.15,791.0,792.0,793.0,794.0,
     > 795.0,796.0,797.0,798.0,799.0,800.0,801.0,802.0,
     > 803.0,804.0,805.0,806.0,807.0,808.0,809.0,810.0,
     > 811.0,812.0,813.0,814.0,815.0,816.0,817.0,818.0,
     > 819.0,820.0,821.0,822.0,823.0,824.0,825.0,826.0,
     > 827.0,828.0,829.0,830.0,831.0,832.0,833.0,834.0,
     > 834.20,835.0,836.0,837.0,838.0,839.0,840.0,841.0,
     > 842.0,843.0,844.0,845.0,846.0,847.0,848.0,849.0,
     > 850.0,851.0,852.0,853.0,854.0,855.0,856.0,857.0,
     > 858.0,859.0,860.0,861.0,862.0,863.0,864.0,865.0,
     > 866.0,867.0,868.0,869.0,870.0,871.0,872.0,873.0,
     > 874.0,875.0,876.0,877.0,878.0,879.0,880.0,881.0,
     > 882.0,883.0,884.0,885.0,886.0,887.0,888.0,889.0,
     > 890.0,891.0,892.0,893.0,894.0,895.0,896.0,897.0,
     > 898.0,899.0,900.0,901.0,902.0,903.0,904.0,904.10,
     > 905.0,906.0,907.0,908.0,909.0,910.0,911.0,912.0,
     > 913.0,914.0,915.0,916.0,917.0,918.0,919.0,920.0,
     > 920.96,921.0,922.0,923.0,923.15,924.0,925.0,926.0,
     > 926.20,927.0,928.0,929.0,930.0,930.75,931.0,932.0,
     > 933.0,933.38,934.0,935.0,936.0,937.0,937.80,938.0,
     > 939.0,940.0,941.0,942.0,943.0,944.0,944.52,945.0,
     > 946.0,947.0,948.0,949.0,949.74,950.0,951.0,952.0,
     > 953.0,954.0,955.0,956.0,957.0,958.0,959.0,960.0,
     > 961.0,962.0,963.0,964.0,965.0,966.0,967.0,968.0,
     > 969.0,970.0,971.0,972.0,972.54,973.0,974.0,975.0,
     > 976.0,977.0,977.02,978.0,979.0,980.0,981.0,982.0,
     > 983.0,984.0,985.0,986.0,987.0,988.0,989.0,989.79,
     > 990.0,991.0,991.55,992.0,993.0,994.0,995.0,996.0,
     > 997.0,998.0,999.0,1000.0,1001.0,1002.0,1003.0,1004.0,
     > 1005.0,1006.0,1007.0,1008.0,1009.0,1010.0,1010.20,1011.0,
     > 1012.0,1013.0,1014.0,1015.0,1016.0,1017.0,1018.0,1019.0,
     > 1020.0,1021.0,1022.0,1023.0,1024.0,1025.0,1025.72,1026.0,
     > 1027.0,1028.0,1029.0,1030.0,1031.0,1031.91,1032.0,1033.0,
     > 1034.0,1035.0,1036.0,1036.34,1037.0,1037.02,1037.61,1038.0,
     > 1039.0,1040.0,1041.0,1042.0,1043.0,1044.0,1045.0,1046.0,
     > 1047.0,1048.0,1049.0,1050.0,1051.0,1052.0,1053.0,1054.0,1055.0/

      DATA F74113/ !.. Fluxes
     > 0.0,0.0,0.0E+00,0.0E+00,1.0E+05,0.0E+00,0.0E+00,
     > 2.0E+05,1.0E+05,6.0E+05,5.0E+05,2.0E+05,5.0E+05,8.0E+05,
     > 3.20E+06,2.80E+06,7.0E+05,1.0E+06,1.80E+06,8.0E+05,2.40E+06,
     > 1.0E+06,1.20E+06,6.0E+05,3.0E+06,3.20E+06,3.60E+06,4.80E+06,
     > 2.0E+05,6.0E+06,6.0E+06,3.20E+06,4.0E+05,5.30E+06,2.30E+06,
     > 2.0E+05,2.80E+06,8.90E+06,2.0E+06,4.80E+06,4.0E+06,3.20E+06,
     > 2.90E+06,1.50E+06,1.40E+06,2.0E+06,2.90E+06,5.20E+06,2.40E+06,
     > 4.0E+06,1.0E+05,1.70E+06,3.0E+06,1.0E+05,2.80E+06,4.50E+06,
     > 3.30E+06,2.0E+05,2.20E+06,2.0E+06,2.40E+06,3.30E+06,2.50E+06,
     > 6.0E+06,2.60E+06,3.10E+06,1.80E+06,1.10E+07,4.0E+05,2.60E+06,
     > 2.40E+06,3.50E+06,3.10E+06,5.10E+06,1.90E+06,1.70E+06,2.70E+06,
     > 2.0E+06,2.0E+06,1.0E+06,3.20E+06,4.10E+06,2.0E+06,3.0E+06,
     > 2.0E+06,5.60E+06,2.0E+06,3.30E+06,2.60E+06,2.30E+06,3.40E+06,
     > 2.40E+06,2.90E+06,1.50E+06,2.80E+06,1.80E+06,2.30E+06,2.30E+06,
     > 3.60E+06,1.60E+06,2.10E+06,2.40E+06,3.40E+06,4.80E+06,6.70E+06,
     > 3.50E+06,4.10E+06,3.50E+06,4.70E+06,3.0E+06,4.70E+06,3.30E+06,
     > 3.10E+06,2.40E+06,4.70E+06,2.20E+06,2.30E+06,1.70E+06,1.40E+06,
     > 5.90E+06,3.60E+06,1.90E+06,1.60E+06,1.0E+05,4.90E+06,1.0E+05,
     > 1.50E+06,1.90E+06,4.50E+06,3.20E+06,3.50E+06,3.70E+06,2.30E+06,
     > 2.40E+06,3.10E+06,1.50E+06,4.60E+06,3.80E+06,3.0E+06,2.40E+06,
     > 3.0E+06,4.50E+06,6.80E+06,1.0E+05,1.40E+06,1.10E+06,1.0E+05,
     > 5.0E+06,2.30E+06,2.30E+06,9.20E+06,1.50E+06,2.60E+06,5.0E+06,
     > 2.40E+06,2.20E+06,4.50E+06,5.20E+06,2.10E+06,1.20E+06,1.30E+06,
     > 1.60E+06,2.20E+06,6.70E+06,2.10E+06,2.70E+06,3.60E+06,1.60E+06,
     > 1.0E+05,3.40E+06,6.0E+06,4.90E+06,1.30E+06,1.60E+06,5.0E+06,
     > 1.80E+06,9.0E+05,1.50E+06,1.50E+06,1.60E+06,1.60E+06,1.0E+05,
     > 1.0E+05,1.0E+05,1.20E+06,1.0E+05,3.80E+06,2.40E+06,2.10E+06,
     > 1.0E+05,1.0E+05,1.90E+06,3.10E+06,2.10E+06,1.0E+05,1.0E+05,
     > 9.0E+05,3.70E+06,2.10E+06,3.60E+06,3.70E+06,1.0E+05,4.20E+06,
     > 3.80E+06,1.0E+05,1.0E+05,1.0E+05,1.0E+05,1.0E+05,9.0E+06,
     > 9.0E+05,1.19E+07,4.0E+07,1.92E+07,1.92E+07,1.28E+07,7.50E+06,
     > 1.20E+07,1.10E+07,6.0E+06,3.20E+06,1.93E+07,3.56E+07,2.02E+07,
     > 1.24E+07,1.87E+07,3.0E+08,9.60E+06,1.71E+07,2.56E+08,2.65E+07,
     > 1.36E+08,1.82E+07,4.0E+05,1.84E+07,2.20E+08,2.50E+07,6.60E+07,
     > 1.69E+07,1.04E+08,6.90E+06,4.45E+07,2.58E+07,7.0E+07,1.90E+06,
     > 1.54E+07,1.91E+08,4.80E+07,1.99E+07,2.25E+07,6.40E+07,7.68E+07,
     > 1.0E+08,1.74E+08,4.53E+07,9.90E+06,1.82E+07,2.33E+07,3.79E+07,
     > 6.60E+07,9.60E+07,2.07E+07,3.90E+07,1.50E+07,9.90E+06,9.0E+05,
     > 9.0E+05,9.0E+05,1.90E+06,1.90E+06,9.0E+05,5.40E+07,1.18E+07,
     > 9.90E+06,7.60E+06,2.17E+07,3.28E+07,5.10E+07,6.50E+07,1.20E+07,
     > 1.70E+07,2.01E+07,1.90E+06,3.74E+07,5.10E+07,9.03E+07,1.90E+06,
     > 2.98E+07,1.84E+07,1.05E+07,1.28E+07,1.87E+07,1.0E+07,8.26E+07,
     > 5.50E+07,1.69E+07,5.40E+07,1.38E+08,9.80E+07,5.0E+07,6.75E+07,
     > 1.49E+07,5.0E+07,1.87E+07,1.99E+07,2.02E+07,9.60E+06,7.30E+07,
     > 3.50E+07,2.90E+07,4.0E+08,6.0E+07,3.0E+07,2.0E+08,2.0E+08,
     > 1.40E+08,5.60E+07,6.20E+07,1.0E+07,6.80E+07,6.40E+07,3.0E+07,
     > 6.0E+07,2.50E+07,9.0E+07,3.0E+07,2.50E+07,5.60E+06,1.0E+07,
     > 4.0E+07,6.0E+07,5.0E+07,1.50E+07,2.10E+08,2.40E+07,1.30E+07,
     > 4.0E+07,1.80E+07,6.34E+07,9.40E+07,9.80E+06,8.0E+08,6.90E+09,
     > 1.20E+08,1.0E+08,2.50E+07,1.30E+08,2.0E+07,1.40E+08,1.0E+08,
     > 8.0E+07,1.50E+08,1.0E+08,1.10E+08,7.0E+07,1.20E+08,6.50E+08,
     > 1.40E+07,3.10E+07,8.20E+07,4.80E+07,0.0E+00,0.0E+00,0.0E+00,
     > 0.0E+00,0.0E+00,0.0E+00,0.0E+00,0.0E+00,0.0E+00,0.0E+00,
     > 0.0E+00,0.0E+00,0.0E+00,2.70E+07,0.0E+00,0.0E+00,0.0E+00,
     > 0.0E+00,0.0E+00,3.0E+05,3.0E+05,3.0E+05,3.0E+05,3.0E+05,
     > 3.0E+05,3.0E+05,3.0E+05,7.40E+07,3.0E+05,3.0E+05,3.0E+05,
     > 3.0E+05,3.0E+05,3.0E+05,1.10E+08,3.0E+05,3.0E+05,3.0E+05,
     > 3.0E+05,3.0E+05,3.0E+05,3.0E+05,7.0E+05,7.0E+05,7.0E+05,
     > 7.0E+05,7.0E+05,7.0E+05,7.0E+05,7.0E+05,7.0E+05,1.0E+06,
     > 1.0E+06,1.0E+06,1.0E+06,1.0E+06,1.0E+06,1.0E+06,1.30E+06,
     > 1.30E+06,1.30E+06,1.30E+06,1.30E+06,1.60E+06,2.90E+08,1.60E+06,
     > 1.60E+06,2.0E+06,2.0E+06,2.0E+06,2.0E+06,2.30E+06,2.30E+06,
     > 2.60E+06,2.60E+06,2.60E+06,3.0E+06,3.0E+06,3.30E+06,3.30E+06,
     > 3.60E+06,3.60E+06,4.0E+06,4.30E+06,4.30E+06,4.60E+06,4.90E+06,
     > 5.30E+06,5.30E+06,1.0E+07,5.60E+06,5.90E+06,6.30E+06,6.60E+06,
     > 6.90E+06,7.60E+06,7.90E+06,8.30E+06,8.60E+06,9.20E+06,1.0E+08,
     > 9.60E+06,1.02E+07,1.06E+07,1.12E+07,1.19E+07,1.10E+08,2.50E+07,
     > 4.60E+07,6.50E+07,1.20E+08,7.20E+08,4.50E+07,7.0E+07,1.27E+09,
     > 1.40E+08,1.0E+05,1.0E+05,5.30E+08,1.0E+05,1.0E+05,1.0E+05,
     > 1.0E+05,1.0E+05,1.0E+05,1.0E+05,1.57E+07,1.0E+05,1.0E+05,
     > 1.0E+05,1.0E+05,1.0E+05,1.0E+05,1.0E+05,2.0E+05,2.40E+08,
     > 2.0E+05,2.0E+05,2.0E+05,2.0E+05,2.0E+05,1.59E+09,2.0E+05,
     > 1.0E+05,1.0E+05,1.0E+05,1.0E+05,1.0E+05,1.0E+05,1.0E+05,
     > 1.0E+05,1.0E+05,1.0E+05,1.09E+07,1.29E+07,1.0E+05,1.69E+07,
     > 1.0E+05,1.0E+05,2.0E+05,2.0E+05,2.0E+05,2.0E+05,2.0E+05,
     > 2.0E+05,2.0E+05,2.0E+05,2.0E+05,2.0E+05,2.0E+05,2.0E+05,
     > 2.0E+05,3.0E+05,1.10E+07,3.0E+05,3.0E+05,3.0E+05,3.0E+05,
     > 1.10E+07,3.0E+05,3.0E+05,3.0E+05,3.0E+05,3.0E+05,4.0E+05,
     > 4.0E+05,4.0E+05,4.0E+05,4.0E+05,4.0E+05,4.0E+05,4.0E+05,
     > 5.0E+05,5.0E+05,5.0E+05,5.0E+05,5.0E+05,5.0E+05,5.0E+05,
     > 5.0E+05,5.0E+05,5.0E+05,5.0E+05,9.0E+07,5.0E+05,6.0E+05,
     > 6.0E+05,6.0E+05,6.0E+05,6.0E+05,7.0E+05,7.0E+05,7.0E+05,
     > 7.0E+05,7.0E+05,7.0E+05,7.0E+05,7.0E+05,8.0E+05,8.0E+05,
     > 8.0E+05,9.0E+05,3.60E+08,9.0E+05,9.0E+05,9.0E+05,9.0E+05,
     > 9.0E+05,9.0E+05,1.0E+06,1.0E+06,1.0E+06,1.10E+06,1.10E+06,
     > 1.20E+06,1.20E+06,1.20E+06,1.30E+06,5.0E+07,1.30E+06,1.30E+06,
     > 1.30E+06,1.40E+06,1.40E+06,1.50E+06,1.50E+06,1.50E+06,1.50E+06,
     > 1.60E+06,1.60E+06,1.70E+06,1.70E+06,1.70E+06,1.90E+06,1.90E+06,
     > 1.90E+06,2.0E+06,2.0E+06,2.10E+06,2.10E+06,2.10E+06,2.20E+06,
     > 2.30E+06,2.30E+06,2.40E+06,2.40E+06,2.50E+06,2.60E+06,2.70E+06,
     > 2.70E+06,2.80E+06,2.90E+06,2.90E+06,3.0E+06,3.10E+06,3.10E+06,
     > 3.20E+06,3.30E+06,3.40E+06,3.0E+07,3.60E+06,2.30E+07,3.60E+06,
     > 8.0E+07,3.70E+06,2.0E+07,3.0E+07,3.80E+06,3.90E+06,4.0E+06,
     > 4.10E+06,1.70E+08,4.20E+06,4.30E+06,4.40E+06,4.50E+06,4.60E+06,
     > 2.60E+08,4.80E+06,4.90E+06,5.0E+06,5.20E+06,5.20E+06,5.40E+06,
     > 1.10E+07,5.60E+06,5.70E+06,5.80E+06,6.0E+06,1.40E+08,6.10E+06,
     > 6.30E+06,6.40E+06,6.60E+06,6.80E+06,6.90E+06,1.30E+08,7.20E+06,
     > 2.50E+08,7.30E+06,7.50E+06,7.60E+06,4.30E+08,7.90E+06,8.10E+06,
     > 8.20E+06,8.50E+06,8.70E+06,8.90E+06,9.10E+06,9.40E+06,9.60E+06,
     > 9.80E+06,1.02E+07,1.04E+07,1.07E+07,1.10E+07,1.12E+07,1.15E+07,
     > 1.18E+07,1.21E+07,1.24E+07,1.27E+07,1.31E+07,1.34E+07,1.37E+07,
     > 1.41E+07,1.45E+07,1.48E+07,1.52E+07,1.55E+07,1.60E+07,1.63E+07,
     > 1.68E+07,1.72E+07,1.77E+07,1.82E+07,1.86E+07,1.91E+07,1.96E+07,
     > 2.0E+07,2.06E+07,2.11E+07,2.16E+07,2.22E+07,2.28E+07,2.33E+07,
     > 6.20E+08,2.40E+07,2.45E+07,2.51E+07,2.58E+07,2.65E+07,2.71E+07,
     > 2.79E+07,2.85E+07,2.93E+07,3.0E+07,3.08E+07,3.16E+07,3.24E+07,
     > 3.32E+07,3.40E+07,3.49E+07,3.58E+07,3.67E+07,3.77E+07,3.86E+07,
     > 3.96E+07,4.06E+07,4.17E+07,4.27E+07,4.38E+07,4.49E+07,4.61E+07,
     > 4.72E+07,4.84E+07,4.96E+07,5.09E+07,5.22E+07,5.35E+07,5.49E+07,
     > 5.63E+07,5.77E+07,5.92E+07,6.07E+07,6.23E+07,6.39E+07,6.55E+07,
     > 6.71E+07,6.88E+07,7.06E+07,7.24E+07,7.42E+07,7.61E+07,7.81E+07,
     > 8.01E+07,8.21E+07,8.42E+07,8.63E+07,8.85E+07,9.08E+07,9.31E+07,
     > 9.55E+07,9.79E+07,1.0E+08,1.03E+08,1.06E+08,1.08E+08,1.11E+08,
     > 1.14E+08,1.17E+08,1.20E+08,1.23E+08,1.26E+08,1.29E+08,1.32E+08,
     > 1.36E+08,1.10E+08,1.39E+08,1.43E+08,1.46E+08,1.50E+08,1.54E+08,
     > 1.58E+08,1.62E+08,1.66E+08,2.30E+06,2.30E+06,2.30E+06,2.40E+06,
     > 2.40E+06,2.50E+06,2.50E+06,2.60E+06,5.60E+07,2.60E+06,2.60E+06,
     > 2.80E+06,8.0E+07,2.80E+06,2.90E+06,2.90E+06,1.10E+08,3.0E+06,
     > 3.0E+06,3.10E+06,3.10E+06,1.30E+08,3.20E+06,3.20E+06,3.30E+06,
     > 1.03E+08,3.30E+06,3.40E+06,3.40E+06,3.50E+06,1.80E+08,3.60E+06,
     > 3.60E+06,3.70E+06,3.70E+06,3.90E+06,4.0E+06,4.0E+06,6.80E+07,
     > 4.10E+06,4.20E+06,4.20E+06,4.30E+06,4.40E+06,3.0E+08,4.40E+06,
     > 4.50E+06,4.60E+06,4.70E+06,4.70E+06,4.80E+06,5.0E+06,5.10E+06,
     > 5.20E+06,5.30E+06,5.30E+06,5.40E+06,5.50E+06,5.60E+06,5.70E+06,
     > 5.80E+06,5.90E+06,6.10E+06,6.20E+06,6.30E+06,6.40E+06,6.50E+06,
     > 6.60E+06,6.0E+08,6.70E+06,6.80E+06,7.0E+06,7.20E+06,7.30E+06,
     > 4.40E+09,7.40E+06,7.50E+06,7.70E+06,7.80E+06,7.90E+06,8.0E+06,
     > 8.30E+06,8.40E+06,8.50E+06,8.70E+06,8.80E+06,9.0E+06,1.70E+08,
     > 9.10E+06,9.40E+06,3.40E+08,9.50E+06,9.70E+06,9.80E+06,1.0E+07,
     > 1.02E+07,1.03E+07,1.06E+07,1.08E+07,1.10E+07,1.11E+07,1.13E+07,
     > 1.16E+07,1.18E+07,1.20E+07,1.22E+07,1.24E+07,1.27E+07,1.29E+07,
     > 1.31E+07,8.0E+07,1.33E+07,1.36E+07,1.39E+07,1.41E+07,1.44E+07,
     > 1.46E+07,1.49E+07,1.52E+07,1.54E+07,1.57E+07,1.60E+07,1.63E+07,
     > 1.66E+07,1.68E+07,1.72E+07,3.50E+09,1.75E+07,1.78E+07,1.82E+07,
     > 1.85E+07,1.88E+07,1.91E+07,2.10E+09,1.95E+07,1.98E+07,2.02E+07,
     > 2.06E+07,2.09E+07,2.0E+08,2.13E+07,2.50E+08,1.04E+09,2.17E+07,
     > 2.21E+07,2.26E+07,2.29E+07,2.33E+07,2.38E+07,2.42E+07,2.46E+07,
     > 2.51E+07,2.55E+07,2.60E+07,2.64E+07,2.70E+07,2.74E+07,2.79E+07,
     > 2.84E+07,2.89E+07,2.95E+07/

	END
C::::::::::::::::::::::::::: GET_PHOTOXS ::::::::::::::
C.. Load Fennelly and Torr photoionization cross sections into
C.. cross section array
C.. Fennelly and Torr photoionization cross sections
C.. Atomic Data and Nuclear Data Tables, 51, 321-363, 1992
C.. P. Richards March 2004
	SUBROUTINE GET_PHOTO_XS(IWLXS,    !.. OUT: # of wavelengths
     >                         WLXS,    !.. OUT: photon wavelengths
     >                           XS)    !.. OUT: cross section array
	IMPLICIT NONE
	INTEGER IWLXS  !.. # of wavelengths
	INTEGER I,K    !.. loop control variables
      !.. Arrays to store cross sections
	REAL WLXS(1571),WLXSD(1571),EV,
     >  OP4S(1571),OP2D(1571),OP2P(1571),OP2E(1571),
     >  OP4E(1571),OP(1571),N2_ABS(1571),N2P(1571),N2P_NP(1571),
     >  N2P_ION(1571),O2_ABS(1571),O2P(1571),O2P_OP(1571),
     >  O2P_ION(1571),NP(1571),XS(15,1571)
        
      IWLXS=1571
      !.. transfer the individual arrays to one array for easier
      !.. calculations
	DO I=1,IWLXS
	  WLXS(I)=WLXSD(I)
        XS(1,I)=OP4S(I)
        XS(2,I)=OP2D(I)
        XS(3,I)=OP2P(I)
        XS(4,I)=OP2E(I)
        XS(5,I)=OP4E(I)
        XS(6,I)=OP(I)
        XS(7,I)=N2_ABS(I)
        XS(8,I)=N2P(I)
        XS(9,I)=N2P_NP(I)
        XS(10,I)=N2P_ION(I)
        XS(11,I)=O2_ABS(I)
        XS(12,I)=O2P(I)
        XS(13,I)=O2P_OP(I) 
        XS(14,I)=O2P_ION(I)
        XS(15,I)=NP(I)
      ENDDO

	DO I=1,IWLXS
        EV=(40.8*303.78)/WLXS(I) !.. wavelength energy
      ENDDO

      !.. Data statements for individual cross sections
      DATA WLXSD 
     > /23.70,28.47,28.79,29.52,30.00,30.27,30.43,31.00,31.62
     > ,33.74,35.40,40.95,41.30,43.76,44.02,44.16,44.70,45.30
     > ,45.66,45.90,46.40,46.67,47.70,47.87,49.22,49.60,50.36
     > ,50.52,50.69,51.70,52.30,52.91,53.90,54.15,54.42,54.70
     > ,55.06,55.34,56.08,56.40,56.92,57.36,57.56,57.88,58.96
     > ,59.62,60.30,60.85,61.07,61.63,61.90,62.30,62.77,62.92
     > ,63.16,63.30,63.65,64.11,64.38,64.60,65.21,65.71,65.85
     > ,66.30,67.14,67.35,67.60,68.35,68.90,69.65,70.00,70.54
     > ,70.75,71.00,71.94,72.19,72.31,72.63,72.80,73.55,74.21
     > ,74.44,74.83,75.03,75.29,75.46,75.73,76.01,76.48,76.83
     > ,76.94,77.30,77.50,77.74,78.56,78.70,79.08,79.48,79.76
     > ,80.00,80.21,80.55,80.94,81.16,81.58,81.94,82.10,82.43
     > ,82.67,83.25,83.42,83.67,84.00,84.26,84.50,84.72,84.86
     > ,85.16,85.50,85.69,85.87,86.23,86.40,86.77,86.98,87.30
     > ,87.61,88.10,88.42,88.60,88.90,89.14,89.70,90.14,90.45
     > ,90.71,91.00,91.48,91.69,91.81,92.09,92.55,92.81,93.61
     > ,94.07,94.25,94.39,94.81,95.37,95.51,95.81,96.05,96.49
     > ,96.83,97.12,97.51,97.87,98.12,98.23,98.50,98.88,99.44
     > ,99.71,99.99,100.54,100.96,101.57,102.15,103.01,103.15,103.30
     > ,103.58,103.94,104.23,104.76,105.23,106.25,106.57,106.93,107.80
     > ,108.05,108.46,109.50,109.98,110.56,110.76,111.16,112.70,113.80
     > ,114.09,114.24,115.39,115.80,116.75,117.20,118.10,120.00,120.40
     > ,121.15,121.79,122.70,123.50,124.00,125.00,127.65,129.87,130.00
     > ,130.30,130.50,131.02,131.21,135.00,136.21,136.45,137.80,140.00
     > ,141.20,144.21,145.00,145.90,148.38,150.00,150.10,152.15,152.84
     > ,154.18,155.00,157.73,158.37,159.94,160.37,164.13,165.00,165.30
     > ,167.50,168.17,168.55,168.92,169.70,170.00,171.06,172.12,172.92
     > ,173.08,174.53,175.00,175.24,175.47,177.10,177.22,178.02,179.27
     > ,179.74,180.00,180.40,180.71,181.14,182.16,182.39,183.45,183.91
     > ,184.10,184.52,184.76,185.00,185.21,186.60,186.87,187.95,188.23
     > ,188.70,190.00,190.70,191.04,191.29,192.38,192.80,193.50,195.00
     > ,195.13,196.52,196.63,197.41,198.53,200.00,201.10,202.05,202.64
     > ,203.78,204.25,204.89,206.26,206.38,206.60,207.46,208.33,209.63
     > ,209.78,209.93,211.32,212.14,213.78,214.75,215.00,215.16,216.88
     > ,217.00,218.19,219.09,220.00,221.26,221.40,221.82,223.26,223.72
     > ,224.74,225.00,225.12,227.01,227.19,227.47,228.70,229.60,230.00
     > ,230.65,231.55,232.60,233.84,234.38,235.00,235.55,237.33,238.40
     > ,239.87,240.00,240.71,241.74,243.03,243.86,244.92,245.94,246.24
     > ,246.91,247.18,248.00,249.18,250.00,251.10,251.95,252.17,253.78
     > ,254.40,255.00,256.32,256.64,257.16,257.39,258.30,259.50,260.00
     > ,261.05,262.99,264.24,264.80,265.00,269.50,270.00,270.50,271.99
     > ,272.64,274.19,275.00,275.35,275.67,276.15,276.77,277.00,277.27
     > ,278.40,280.00,280.90,281.41,284.15,285.00,285.70,285.85,288.36
     > ,289.17,290.00,290.69,291.63,292.00,292.78,295.00,295.57,296.17
     > ,299.50,300.00,300.80,303.31,303.78,310.00,315.00,316.10,316.20
     > ,319.01,319.83,320.00,320.56,325.00,330.00,335.00,335.39,340.00
     > ,345.00,345.13,345.74,347.39,349.85,350.00,353.86,356.01,360.00
     > ,360.76,364.48,364.80,368.07,370.00,374.74,380.00,390.00,399.82
     > ,400.00,401.14,401.70,401.94,403.26,405.00,406.00,407.00,408.00
     > ,409.00,410.00,411.00,412.00,413.00,414.00,415.00,416.00,417.00
     > ,417.24,417.71,418.00,419.00,420.00,421.00,422.00,423.00,424.00
     > ,425.00,426.00,427.00,428.00,429.00,430.00,430.47,431.00,432.00
     > ,433.00,434.00,435.00,436.00,436.10,436.67,437.00,438.00,439.00
     > ,440.00,440.50,441.00,442.00,442.30,442.70,443.00,443.60,444.00
     > ,445.00,445.10,446.00,446.20,446.70,447.00,447.60,448.00,448.60
     > ,449.00,450.00,450.20,451.00,452.00,452.60,453.00,454.00,454.30
     > ,454.70,455.00,455.40,456.00,456.50,457.00,458.00,459.00,459.20
     > ,460.00,461.00,462.00,463.00,464.00,465.00,465.22,466.00,467.00
     > ,468.00,469.00,469.80,470.00,471.00,472.00,473.00,473.20,473.80
     > ,474.00,475.00,475.90,476.00,476.40,477.00,478.00,478.50,479.00
     > ,480.00,480.80,481.00,481.70,482.00,482.10,482.60,483.00,483.30
     > ,484.00,484.70,485.00,486.00,486.60,487.00,488.00,489.00,489.50
     > ,490.00,491.00,491.70,492.00,493.00,494.00,495.00,496.00,497.00
     > ,498.00,499.00,499.27,499.37,500.00,501.00,501.10,502.00,503.00
     > ,504.00,505.00,507.90,510.00,515.60,520.00,520.66,521.10,525.80
     > ,530.00,537.02,540.00,542.80,544.70,548.90,550.00,551.40,554.37
     > ,554.51,555.60,558.60,560.00,562.80,565.00,568.50,570.00,572.30
     > ,575.00,580.00,580.40,584.00,584.33,585.00,585.80,588.90,590.00
     > ,592.40,594.10,595.00,596.70,599.60,600.00,604.30,608.00,608.40
     > ,608.90,609.70,610.00,610.70,610.90,612.00,612.40,612.70,613.00
     > ,614.00,615.00,615.20,616.00,616.30,616.60,616.90,617.70,618.00
     > ,618.20,618.70,618.90,619.80,620.00,620.50,621.00,621.90,623.00
     > ,624.00,624.50,624.93,625.90,626.60,627.00,628.00,629.00,629.60
     > ,629.70,630.00,630.30,631.00,631.60,632.00,633.00,634.00,634.40
     > ,634.70,635.00,635.80,636.00,636.30,637.00,637.30,638.00,638.50
     > ,638.70,639.00,640.00,640.41,640.93,641.81,642.00,642.50,643.00
     > ,644.00,644.20,644.40,645.00,645.90,646.60,646.70,647.00,647.50
     > ,648.00,649.00,649.40,650.00,650.30,651.00,651.80,652.00,653.00
     > ,654.00,655.00,656.00,657.00,657.30,658.00,659.00,660.00,660.30
     > ,661.00,661.40,661.90,663.00,664.00,664.60,664.90,665.30,665.80
     > ,666.00,667.00,667.30,668.00,669.00,669.60,670.00,670.40,671.00
     > ,671.50,671.90,672.90,673.60,673.80,674.00,674.40,675.00,675.20
     > ,675.70,676.00,676.20,676.60,677.00,677.50,677.90,678.30,678.80
     > ,679.00,679.20,679.90,680.20,680.40,680.70,681.00,681.30,681.40
     > ,681.60,681.70,682.00,682.30,682.80,683.00,683.30,683.80,683.90
     > ,684.30,684.50,684.80,684.90,685.50,685.71,686.00,686.30,686.60
     > ,686.70,687.00,687.30,687.90,688.40,688.70,689.00,690.00,690.60
     > ,690.80,691.00,691.20,691.40,692.00,692.20,692.40,692.70,693.00
     > ,693.80,694.00,694.30,694.90,695.20,696.00,696.50,697.00,697.30
     > ,697.50,697.70,698.00,698.30,698.90,699.40,699.60,699.70,700.00
     > ,700.30,700.40,700.80,701.00,701.20,701.60,701.80,702.00,703.00
     > ,703.36,703.60,704.00,704.50,705.00,705.30,705.90,706.60,707.00
     > ,707.60,707.90,708.90,709.20,709.70,710.00,710.80,711.00,711.90
     > ,712.50,712.70,712.90,713.50,713.90,714.70,715.00,715.60,715.70
     > ,716.00,716.50,717.00,717.20,717.60,718.00,718.50,719.00,719.20
     > ,719.40,720.00,720.40,720.90,721.30,721.40,721.80,722.00,722.50
     > ,722.90,723.40,723.90,724.20,724.80,724.90,725.50,725.70,726.00
     > ,726.40,727.00,727.30,727.50,728.00,728.30,728.70,729.00,729.40
     > ,729.80,730.00,730.60,730.90,731.50,731.80,732.00,732.20,732.50
     > ,733.00,733.30,734.00,734.50,735.00,735.30,735.90,736.50,737.00
     > ,737.20,737.50,738.00,738.40,739.00,739.20,740.00,740.20,741.00
     > ,741.20,742.00,742.20,742.40,743.00,743.20,743.50,743.70,744.00
     > ,744.50,744.90,746.00,746.40,746.70,747.00,747.50,748.00,748.50
     > ,749.00,749.20,750.00,750.70,751.00,751.60,752.00,752.30,752.90
     > ,754.00,754.40,755.00,755.20,755.50,755.80,756.00,756.20,756.50
     > ,757.00,757.20,757.50,757.70,758.00,758.30,758.40,758.68,759.00
     > ,759.40,760.00,760.20,760.40,760.70,761.00,761.50,762.00,762.20
     > ,762.50,763.00,763.20,763.50,763.70,764.00,764.40,764.70,765.00
     > ,765.30,765.40,765.70,766.00,766.50,766.70,767.00,767.30,767.50
     > ,767.70,767.90,768.30,768.40,768.70,769.00,769.20,769.50,770.00
     > ,770.20,770.40,770.80,771.00,771.50,772.00,772.40,773.00,773.50
     > ,774.00,774.50,775.00,775.50,775.70,776.00,776.50,777.00,777.50
     > ,778.00,778.50,778.70,779.00,779.50,779.80,779.90,780.30,780.50
     > ,781.00,781.20,781.50,782.00,782.50,782.90,783.20,783.50,783.80
     > ,784.00,784.40,784.80,785.00,785.50,786.00,786.20,786.40,787.00
     > ,787.50,787.70,788.00,788.50,789.00,789.50,790.00,790.50,790.80
     > ,791.00,791.30,791.40,791.80,792.40,792.80,792.92,793.14,793.50
     > ,794.00,794.50,795.00,795.20,796.00,797.00,797.70,798.00,799.00
     > ,799.50,800.00,800.50,801.00,801.50,802.00,802.60,803.00,803.50
     > ,804.00,804.27,804.38,804.50,804.78,805.00,805.29,805.44,805.74
     > ,806.00,806.23,806.42,807.00,808.00,808.20,808.50,808.80,809.00
     > ,809.30,809.50,810.00,810.50,810.66,810.85,811.00,811.26,811.49
     > ,811.61,811.80,812.00,812.27,812.50,812.80,813.00,813.50,813.70
     > ,814.00,814.50,814.90,815.50,816.00,816.42,816.77,817.00,817.19
     > ,817.50,817.78,818.00,818.20,818.34,818.50,819.00,819.50,819.80
     > ,820.00,820.50,821.00,821.30,821.50,822.00,823.00,823.20,824.00
     > ,824.50,824.90,825.30,825.50,826.00,826.50,826.80,827.00,827.50
     > ,827.80,828.00,828.30,829.00,829.40,829.60,829.80,830.00,831.00
     > ,832.00,832.50,832.80,832.90,833.50,833.70,834.00,834.20,834.50
     > ,835.00,835.20,835.40,836.00,836.30,837.00,837.50,837.80,838.00
     > ,838.60,838.90,840.00,840.50,840.70,841.00,841.50,842.00,842.50
     > ,843.00,843.50,843.80,844.00,844.50,845.00,845.50,845.90,847.00
     > ,847.60,848.00,848.50,849.00,849.20,849.50,850.00,850.60,851.00
     > ,851.50,851.80,852.00,852.19,852.50,853.00,853.20,853.50,854.00
     > ,854.50,855.00,855.50,856.00,856.20,856.50,857.00,857.30,857.70
     > ,858.00,858.50,859.00,859.20,860.00,860.40,861.00,861.50,862.00
     > ,863.00,863.20,864.00,864.60,865.00,865.20,865.40,866.00,866.50
     > ,867.00,867.50,868.00,868.50,869.00,869.50,870.00,870.80,871.00
     > ,871.40,872.00,872.50,873.00,873.50,874.00,874.50,875.00,875.20
     > ,875.50,876.00,876.20,876.50,877.00,877.50,877.72,878.50,878.92
     > ,879.20,879.50,879.80,880.00,880.50,881.00,881.50,882.00,882.50
     > ,882.70,883.00,883.30,883.50,884.00,884.50,885.00,885.50,885.80
     > ,886.00,886.50,886.90,887.50,888.00,888.50,889.00,889.50,890.00
     > ,890.50,891.00,891.50,892.00,892.50,893.00,893.50,894.00,894.50
     > ,894.70,895.00,895.50,895.80,896.00,896.50,897.00,897.20,897.50
     > ,898.00,898.50,898.70,899.00,899.50,900.00,900.20,900.50,901.00
     > ,901.30,901.50,902.00,903.00,903.50,903.80,904.00,904.50,905.00
     > ,905.50,906.00,906.40,907.00,907.40,908.00,908.50,909.00,909.50
     > ,909.80,910.00,910.40,911.00,911.50,911.76,912.00,912.32,912.50
     > ,913.00,913.30,913.50,914.00,914.50,914.70,915.00,915.50,916.00
     > ,916.50,917.00,917.20,917.50,918.00,918.50,918.90,919.50,919.90
     > ,920.40,920.96,921.50,922.00,922.50,922.80,923.00,923.50,923.70
     > ,924.00,924.30,924.50,925.00,925.50,926.00,926.20,926.40,927.00
     > ,927.60,928.00,928.50,929.00,930.00,930.50,930.75,931.00,931.50
     > ,931.90,932.40,933.00,933.38,933.50,934.00,934.50,935.00,935.30
     > ,935.50,936.00,936.50,937.00,937.50,937.80,937.90,938.50,939.00
     > ,939.30,939.50,940.00,940.50,941.00,941.50,942.00,942.40,943.00
     > ,943.30,943.50,944.00,944.50,945.00,946.00,947.00,947.70,948.00
     > ,948.50,949.00,949.50,949.74,950.00,950.30,950.50,951.00,952.00
     > ,953.00,954.00,955.00,955.90,956.50,956.70,957.00,957.50,958.00
     > ,958.20,958.50,958.80,959.00,959.50,960.00,960.50,961.00,961.50
     > ,961.90,962.50,962.80,963.00,964.00,965.00,965.50,966.00,966.50
     > ,967.00,967.50,968.00,969.00,969.50,970.00,970.40,971.00,971.50
     > ,972.00,972.50,972.90,973.50,974.00,974.50,975.00,975.30,975.50
     > ,976.00,976.50,977.00,977.50,978.00,978.50,979.00,979.50,980.00
     > ,980.50,981.00,981.50,982.00,982.50,983.00,983.30,983.50,984.00
     > ,984.50,985.00,985.20,985.50,985.90,986.30,987.00,988.00,988.50
     > ,989.00,989.60,989.79,990.00,991.00,991.50,992.00,992.90,993.20
     > ,993.50,994.00,995.00,996.00,997.00,997.20,998.00,999.00,1000.00
     > ,1001.00,1002.00,1003.00,1004.00,1004.30,1004.60,1005.00,1006.00
     > ,1006.80,1007.00,1007.90,1009.00,1009.40,1010.00,1010.20,1011.00
     > ,1011.40,1012.00,1012.30,1013.00,1013.50,1013.90,1015.00,1015.80
     > ,1016.00,1016.40,1016.90,1017.20,1017.80,1018.00,1018.30,1018.80
     > ,1019.00,1019.40,1020.00,1020.40,1020.80,1021.00,1021.60,1022.00
     > ,1022.40,1023.00,1023.40,1024.00,1024.60,1025.00,1025.30,1025.70
     > ,1026.00,1027.00/

       DATA OP4S/0.01,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.03
     > ,0.03,0.04,0.04,0.05,0.05,0.05,0.05,0.05,0.05,0.05
     > ,0.05,0.05,0.05,0.05,0.06,0.05,0.05,0.05,0.06,0.06
     > ,0.06,0.06,0.06,0.06,0.07,0.06,0.07,0.07,0.07,0.07
     > ,0.07,0.07,0.07,0.08,0.08,0.08,0.09,0.09,0.09,0.09
     > ,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.11,0.11,0.11
     > ,0.11,0.12,0.12,0.12,0.12,0.12,0.12,0.13,0.13,0.13
     > ,0.14,0.14,0.14,0.15,0.15,0.15,0.15,0.15,0.16,0.15
     > ,0.16,0.17,0.16,0.17,0.17,0.17,0.17,0.17,0.17,0.18
     > ,0.18,0.18,0.18,0.18,0.18,0.19,0.19,0.20,0.20,0.20
     > ,0.20,0.20,0.21,0.21,0.21,0.21,0.21,0.22,0.22,0.22
     > ,0.22,0.22,0.23,0.23,0.23,0.23,0.23,0.23,0.23,0.24
     > ,0.24,0.24,0.24,0.24,0.24,0.24,0.25,0.25,0.25,0.25
     > ,0.25,0.26,0.26,0.26,0.27,0.26,0.27,0.27,0.27,0.27
     > ,0.28,0.28,0.28,0.29,0.29,0.29,0.29,0.30,0.30,0.30
     > ,0.30,0.30,0.31,0.31,0.31,0.31,0.32,0.32,0.32,0.32
     > ,0.32,0.33,0.33,0.33,0.33,0.34,0.34,0.34,0.35,0.35
     > ,0.35,0.35,0.36,0.36,0.36,0.36,0.37,0.37,0.38,0.38
     > ,0.38,0.39,0.39,0.40,0.40,0.40,0.41,0.42,0.42,0.43
     > ,0.43,0.44,0.44,0.44,0.45,0.45,0.47,0.47,0.48,0.48
     > ,0.49,0.49,0.50,0.50,0.53,0.55,0.55,0.55,0.55,0.56
     > ,0.56,0.58,0.58,0.59,0.60,0.60,0.59,0.59,0.59,0.59
     > ,0.59,0.59,0.58,0.58,0.56,0.56,0.56,0.61,0.62,0.65
     > ,0.66,0.73,0.74,0.75,0.79,0.80,0.81,0.81,0.83,0.83
     > ,0.85,0.87,0.85,0.86,0.91,0.90,0.89,0.90,0.93,0.93
     > ,0.93,0.95,0.95,0.95,0.96,0.96,0.96,0.97,0.98,0.98
     > ,0.99,0.99,0.99,1.00,1.00,1.00,1.01,1.02,1.03,1.03
     > ,1.03,1.05,1.05,1.05,1.06,1.07,1.07,1.08,1.09,1.09
     > ,1.11,1.11,1.11,1.12,1.14,1.11,1.11,1.12,1.13,1.13
     > ,1.14,1.15,1.15,1.15,1.16,1.17,1.18,1.18,1.18,1.19
     > ,1.20,1.21,1.22,1.22,1.23,1.24,1.24,1.25,1.26,1.27
     > ,1.28,1.28,1.28,1.29,1.30,1.31,1.31,1.31,1.32,1.33
     > ,1.33,1.34,1.35,1.35,1.36,1.36,1.37,1.38,1.39,1.39
     > ,1.40,1.41,1.42,1.43,1.43,1.44,1.42,1.40,1.41,1.42
     > ,1.43,1.49,1.49,1.49,1.50,1.51,1.52,1.53,1.48,1.54
     > ,1.55,1.56,1.56,1.57,1.58,1.58,1.58,1.59,1.53,1.54
     > ,1.55,1.56,1.57,1.58,1.58,1.62,1.63,1.63,1.64,1.65
     > ,1.66,1.67,1.67,1.67,1.68,1.68,1.69,1.69,1.70,1.71
     > ,1.72,1.73,1.77,1.78,1.79,1.79,1.83,1.84,1.85,1.85
     > ,1.86,1.86,1.86,1.87,1.87,1.87,1.89,1.89,1.90,1.92
     > ,1.92,2.02,2.10,2.11,2.11,2.14,2.14,2.14,2.15,2.19
     > ,2.24,2.28,2.28,2.32,2.36,2.36,2.37,2.38,2.40,2.40
     > ,2.44,2.46,2.49,2.49,2.53,2.53,2.56,2.57,2.59,2.65
     > ,2.73,3.06,3.07,3.04,3.03,3.03,3.00,2.96,2.94,2.92
     > ,2.90,2.88,2.86,2.87,2.87,2.88,2.89,2.90,2.90,2.91
     > ,2.91,2.92,2.92,2.93,2.94,2.94,2.95,2.95,2.96,2.96
     > ,2.97,2.97,2.98,2.98,2.99,2.99,2.99,2.98,2.98,2.97
     > ,2.97,3.19,3.19,3.19,3.20,3.25,3.29,3.34,3.36,3.27
     > ,3.10,3.05,3.61,3.49,3.25,3.21,3.12,3.11,2.97,2.94
     > ,3.92,3.74,3.39,3.28,3.11,3.12,3.16,3.16,3.14,3.10
     > ,3.08,3.01,2.85,2.80,3.08,3.57,4.23,3.88,3.52,3.41
     > ,3.31,3.21,3.21,3.31,3.31,3.32,3.33,3.33,3.34,3.34
     > ,3.34,3.35,3.36,3.36,3.37,3.37,3.37,3.38,3.39,3.39
     > ,3.36,3.35,3.30,3.22,3.21,3.16,3.25,3.40,3.47,3.45
     > ,3.42,4.51,4.36,3.84,3.78,3.76,3.67,3.65,3.64,3.63
     > ,3.61,3.60,3.56,3.54,3.52,3.47,3.41,3.39,3.36,3.40
     > ,3.44,3.43,3.42,3.41,3.39,3.38,3.37,3.36,3.34,3.34
     > ,3.34,3.33,3.30,3.29,3.30,3.30,3.31,3.31,3.32,3.33
     > ,3.35,3.36,3.36,3.36,3.38,3.43,3.55,3.57,3.58,3.59
     > ,3.62,3.63,3.63,3.65,3.65,3.66,3.67,3.68,3.70,3.71
     > ,3.73,3.74,3.75,3.76,3.77,3.79,3.92,3.92,3.90,3.88
     > ,3.79,3.76,3.69,3.64,3.61,3.56,3.47,3.46,3.53,3.59
     > ,3.59,3.60,3.61,3.62,3.63,3.63,3.64,3.65,3.65,3.66
     > ,3.67,3.68,3.69,3.70,3.70,3.71,3.71,3.72,3.73,3.73
     > ,3.73,3.74,3.75,3.75,3.76,3.77,3.78,3.79,3.81,3.81
     > ,3.82,3.83,3.84,3.85,3.86,3.87,3.88,3.88,3.89,3.95
     > ,4.11,4.25,4.34,4.56,4.78,4.87,4.94,5.01,5.18,5.23
     > ,5.30,5.45,5.52,5.67,5.79,5.83,5.90,6.12,6.01,5.88
     > ,5.66,5.61,5.49,5.36,5.11,5.06,5.01,4.87,4.64,4.47
     > ,4.45,4.37,4.25,4.13,3.88,3.79,3.64,3.63,3.60,3.57
     > ,3.57,3.53,3.49,3.46,3.42,3.38,3.37,3.35,3.31,3.28
     > ,3.31,3.40,3.44,3.50,3.63,3.74,3.81,3.84,4.16,4.67
     > ,4.87,4.77,4.74,4.66,4.56,4.50,4.46,4.42,4.36,4.31
     > ,4.27,4.17,4.10,4.08,4.06,4.02,3.96,3.94,3.90,3.87
     > ,3.85,3.81,3.77,3.73,3.82,3.92,3.96,5.48,4.47,4.43
     > ,4.41,4.39,4.61,4.83,5.05,5.13,4.93,4.82,4.50,4.16
     > ,3.56,3.59,3.60,3.62,3.62,3.88,3.92,4.03,4.58,6.94
     > ,6.94,6.28,5.55,4.79,4.53,3.98,3.41,3.95,4.06,3.97
     > ,3.88,3.52,3.52,3.56,3.60,3.64,3.68,3.80,3.81,3.85
     > ,3.84,3.79,3.68,3.69,3.71,3.75,3.77,3.74,3.96,3.24
     > ,2.75,9.67,7.42,7.65,7.59,6.58,6.34,6.14,6.01,5.54
     > ,4.56,4.24,3.82,3.61,3.40,5.24,6.14,5.98,5.17,4.89
     > ,4.70,4.39,4.15,3.94,3.96,4.02,4.07,4.10,4.15,4.15
     > ,4.15,4.14,4.01,3.93,4.03,4.03,4.06,4.07,4.08,4.08
     > ,4.09,4.10,4.13,4.15,4.18,4.18,4.16,4.13,4.09,4.08
     > ,4.05,4.09,4.13,4.18,4.20,4.21,4.23,4.10,3.95,3.85
     > ,3.83,3.73,3.68,3.56,3.41,3.61,3.68,9.72,12.23,9.23
     > ,6.59,5.48,5.16,4.62,3.56,2.92,2.45,2.66,2.78,2.95
     > ,3.08,3.16,3.24,3.27,3.37,3.44,3.58,3.65,8.08,8.03
     > ,6.97,5.21,4.15,8.62,11.81,15.00,15.34,16.01,16.69,10.40
     > ,12.30,15.15,12.50,8.90,14.37,16.20,9.08,7.30,18.41,18.91
     > ,11.77,9.98,8.20,6.33,5.71,10.08,14.75,21.76,17.10,10.10
     > ,5.42,4.85,4.70,5.49,6.80,20.10,33.40,20.33,15.10,9.82
     > ,5.20,5.00,4.61,4.35,4.15,4.36,4.76,4.90,22.34,28.16
     > ,36.88,45.60,40.19,34.77,26.65,13.11,7.70,9.57,10.83,12.70
     > ,11.38,10.94,9.71,8.30,6.54,3.90,4.05,4.06,4.05,4.05
     > ,4.04,4.04,4.04,4.03,4.03,4.02,4.02,4.02,4.01,4.01
     > ,4.00,4.00,3.99,3.99,3.99,3.98,3.97,3.97,3.96,3.95
     > ,3.95,3.95,3.94,3.93,3.93,3.93,3.92,3.92,3.91,3.90
     > ,3.90,3.89,3.88,3.88,3.87,3.86,3.85,3.84,3.83,3.82
     > ,3.81,3.80,3.79,3.79,3.78,3.77,3.76,3.75,3.74,3.73
     > ,3.73,3.72,3.71,3.70,3.70,3.70,3.70,3.70,3.70,3.70
     > ,3.70,3.70,3.70,3.70,3.70,3.70,3.70,3.70,3.70,3.70
     > ,3.71,3.71,3.72,3.72,3.72,3.73,3.73,3.74,3.74,3.75
     > ,3.76,3.76,3.77,3.77,3.77,3.78,3.78,4.23,3.78,4.86
     > ,19.97,2.70,2.74,2.80,2.86,2.92,2.94,3.03,3.15,3.23
     > ,3.26,3.38,3.44,3.50,3.55,3.61,3.67,3.73,3.80,3.84
     > ,3.90,3.96,36.41,9.14,5.34,48.94,7.20,11.73,3.64,15.37
     > ,3.64,5.91,1.62,2.06,2.13,2.14,2.17,2.19,2.20,2.22
     > ,2.24,2.27,2.31,15.23,2.82,32.62,3.39,7.90,4.74,2.80
     > ,3.67,3.95,1.97,2.21,2.38,2.79,2.95,3.19,3.60,3.93
     > ,4.42,4.83,5.17,65.14,4.14,1.86,3.10,22.02,2.59,7.40
     > ,1.96,1.55,1.68,1.82,1.90,1.95,2.09,2.22,2.30,2.36
     > ,2.49,2.76,2.82,3.03,3.17,3.27,3.30,3.30,3.30,3.30
     > ,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30
     > ,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30
     > ,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30
     > ,3.30,3.30,3.30,3.30,3.29,3.29,3.29,3.28,3.28,3.27
     > ,3.27,3.26,3.26,3.26,3.25,3.25,3.24,3.23,3.22,3.22
     > ,3.22,3.21,3.21,3.21,3.20,3.19,3.18,3.17,3.16,3.16
     > ,3.16,3.15,3.14,3.14,3.13,3.12,3.11,3.10,3.10,3.10
     > ,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10
     > ,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10
     > ,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10
     > ,3.77,3.94,4.27,4.77,5.19,5.61,6.02,6.44,6.86,7.28
     > ,7.44,7.70,8.11,8.28,8.53,8.95,9.37,11.82,34.88,9.93
     > ,4.73,6.81,5.20,5.11,4.88,4.66,4.43,4.21,3.98,3.89
     > ,3.75,3.62,3.53,3.30,3.08,2.85,2.84,2.83,2.82,2.81
     > ,2.79,2.78,2.76,2.75,2.73,2.72,2.70,2.71,2.71,2.72
     > ,2.72,2.73,2.73,2.74,2.74,2.75,2.75,2.75,2.76,2.77
     > ,2.77,2.78,2.79,2.79,2.80,2.81,2.82,2.82,2.83,2.84
     > ,2.85,2.85,2.84,2.83,2.82,2.82,2.81,2.79,2.78,2.77
     > ,2.77,2.76,2.75,2.75,2.75,2.75,2.75,2.75,2.68,2.62
     > ,2.55,2.49,2.45,1.15,0.75,0.49,0.30,0.20,0.14,0.06
     > ,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA OP2D/0.01,0.02,0.02,0.02,0.02,0.02,0.03,0.03,0.03,0.03
     > ,0.03,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05
     > ,0.05,0.05,0.05,0.05,0.06,0.06,0.06,0.06,0.06,0.07
     > ,0.07,0.07,0.07,0.08,0.08,0.08,0.08,0.08,0.08,0.08
     > ,0.09,0.09,0.09,0.09,0.10,0.10,0.10,0.10,0.11,0.11
     > ,0.11,0.11,0.11,0.11,0.11,0.12,0.12,0.12,0.12,0.12
     > ,0.12,0.13,0.13,0.13,0.14,0.13,0.14,0.14,0.15,0.15
     > ,0.15,0.16,0.15,0.16,0.16,0.16,0.16,0.16,0.16,0.17
     > ,0.17,0.18,0.18,0.18,0.18,0.18,0.18,0.19,0.18,0.19
     > ,0.19,0.19,0.20,0.20,0.20,0.20,0.21,0.21,0.21,0.21
     > ,0.22,0.21,0.22,0.22,0.23,0.23,0.23,0.23,0.23,0.23
     > ,0.24,0.24,0.24,0.24,0.24,0.25,0.25,0.25,0.25,0.25
     > ,0.25,0.25,0.26,0.26,0.26,0.26,0.26,0.27,0.27,0.27
     > ,0.27,0.27,0.28,0.28,0.28,0.28,0.29,0.29,0.29,0.29
     > ,0.30,0.30,0.30,0.31,0.31,0.31,0.31,0.32,0.32,0.32
     > ,0.32,0.32,0.33,0.33,0.33,0.33,0.34,0.34,0.34,0.34
     > ,0.34,0.35,0.35,0.35,0.36,0.36,0.36,0.37,0.37,0.37
     > ,0.37,0.38,0.38,0.38,0.39,0.39,0.40,0.40,0.40,0.41
     > ,0.41,0.41,0.42,0.42,0.43,0.43,0.43,0.44,0.45,0.46
     > ,0.46,0.46,0.47,0.47,0.48,0.48,0.50,0.50,0.51,0.51
     > ,0.52,0.52,0.53,0.54,0.56,0.58,0.59,0.59,0.59,0.59
     > ,0.60,0.63,0.64,0.65,0.66,0.66,0.66,0.65,0.67,0.67
     > ,0.65,0.66,0.67,0.66,0.66,0.66,0.66,0.72,0.73,0.77
     > ,0.77,0.85,0.87,0.88,0.93,0.94,0.95,0.96,0.97,0.98
     > ,1.00,1.03,1.04,1.05,1.08,1.09,1.09,1.10,1.13,1.13
     > ,1.14,1.16,1.16,1.16,1.17,1.21,1.18,1.23,1.23,1.24
     > ,1.25,1.21,1.25,1.26,1.26,1.26,1.28,1.28,1.29,1.30
     > ,1.30,1.32,1.32,1.33,1.33,1.34,1.35,1.36,1.37,1.38
     > ,1.39,1.39,1.40,1.42,1.43,1.45,1.46,1.46,1.48,1.48
     > ,1.49,1.51,1.51,1.51,1.52,1.53,1.54,1.54,1.55,1.56
     > ,1.57,1.59,1.60,1.60,1.60,1.62,1.62,1.64,1.65,1.66
     > ,1.72,1.72,1.72,1.69,1.71,1.76,1.76,1.76,1.78,1.79
     > ,1.79,1.80,1.81,1.82,1.82,1.83,1.85,1.86,1.87,1.87
     > ,1.88,1.90,1.91,1.93,1.93,1.94,1.95,1.96,1.97,1.98
     > ,2.00,2.00,2.01,2.01,2.04,2.09,2.10,2.12,2.13,2.13
     > ,2.15,2.16,2.17,2.18,2.18,2.19,2.19,2.20,2.21,2.21
     > ,2.23,2.25,2.27,2.27,2.28,2.33,2.34,2.35,2.37,2.37
     > ,2.39,2.40,2.41,2.41,2.42,2.43,2.43,2.43,2.45,2.47
     > ,2.48,2.49,2.62,2.64,2.65,2.65,2.70,2.72,2.74,2.74
     > ,2.75,2.75,2.75,2.77,2.77,2.77,2.79,2.79,2.81,2.84
     > ,2.85,3.01,3.15,3.16,3.17,3.20,3.22,3.22,3.23,3.31
     > ,3.41,3.50,3.51,3.57,3.63,3.64,3.64,3.67,3.70,3.70
     > ,3.75,3.78,3.83,3.84,3.89,3.89,3.93,3.97,4.00,4.12
     > ,4.27,4.82,4.83,4.80,4.78,4.77,4.73,4.67,4.64,4.61
     > ,4.58,4.54,4.51,4.41,4.53,4.54,4.56,4.57,4.58,4.59
     > ,4.60,4.60,4.61,4.73,4.63,4.64,4.76,4.77,4.78,4.79
     > ,4.80,4.80,4.81,4.82,4.83,4.83,4.82,4.81,4.81,4.80
     > ,4.79,5.13,5.13,5.12,5.15,5.22,5.29,5.36,5.40,5.26
     > ,4.99,4.90,5.80,5.61,5.22,5.16,5.01,4.99,4.77,4.72
     > ,6.30,6.02,5.45,5.27,4.99,5.02,5.07,5.09,5.04,4.98
     > ,4.95,4.84,4.58,4.47,4.87,5.61,6.70,6.23,5.76,5.68
     > ,5.51,5.34,5.31,5.32,5.33,5.34,5.35,5.36,5.37,5.37
     > ,5.38,5.39,5.39,5.40,5.41,5.41,5.42,5.43,5.44,5.45
     > ,5.40,5.38,5.31,5.17,5.16,5.09,5.23,5.46,5.58,5.55
     > ,5.49,7.24,7.00,6.16,6.08,6.05,5.89,5.87,5.85,5.83
     > ,5.80,5.79,5.73,5.69,5.66,5.57,5.49,5.44,5.40,5.47
     > ,5.52,5.52,5.50,5.48,5.46,5.44,5.42,5.40,5.38,5.37
     > ,5.37,5.35,5.30,5.29,5.30,5.31,5.31,5.32,5.34,5.35
     > ,5.38,5.40,5.40,5.40,5.43,5.45,5.51,5.53,5.56,5.58
     > ,5.62,5.63,5.64,5.66,5.67,5.68,5.70,5.71,5.74,5.76
     > ,5.79,5.80,5.82,5.78,5.72,5.75,6.02,6.02,6.03,6.03
     > ,6.06,6.07,6.08,6.09,6.10,6.10,6.12,6.12,6.14,6.15
     > ,6.16,6.16,6.16,6.16,6.16,6.16,6.16,6.16,6.16,6.16
     > ,6.16,6.16,6.16,6.16,6.16,6.16,6.16,6.16,6.16,6.16
     > ,6.16,6.16,6.16,6.16,6.17,6.18,6.19,6.20,6.22,6.22
     > ,6.23,6.24,6.25,6.26,6.27,6.28,6.29,6.29,6.30,6.29
     > ,6.27,6.25,6.24,6.20,6.17,6.16,6.15,6.14,6.12,6.11
     > ,6.10,6.08,6.07,6.05,6.03,6.03,6.02,5.99,5.97,5.96
     > ,5.94,5.93,5.92,5.90,5.88,5.87,5.87,5.85,5.83,5.81
     > ,5.81,5.80,5.79,5.77,5.75,5.74,5.72,5.72,5.72,5.71
     > ,5.71,5.71,5.70,5.70,5.69,5.69,5.68,5.68,5.68,5.67
     > ,5.65,5.59,5.56,5.53,5.44,5.37,5.32,5.30,5.78,6.58
     > ,6.89,6.85,6.83,6.80,6.75,6.72,6.70,6.67,6.64,6.62
     > ,6.59,6.54,6.50,6.49,6.48,6.45,6.42,6.41,6.38,6.36
     > ,6.35,6.32,6.30,6.27,6.47,6.68,8.04,12.19,10.93,10.09
     > ,9.74,9.51,9.68,9.85,10.02,10.07,9.49,9.20,8.34,7.50
     > ,6.13,6.06,5.92,5.68,5.63,5.82,6.28,7.17,8.55,17.74
     > ,18.55,16.29,13.98,11.72,10.97,9.37,7.79,8.52,8.35,7.95
     > ,7.56,6.28,5.95,5.92,5.88,5.85,5.81,5.70,5.70,5.72
     > ,5.67,5.57,5.32,5.31,5.31,5.30,5.29,5.17,4.84,6.12
     > ,6.95,30.33,29.67,25.06,20.75,13.02,9.83,8.66,8.10,6.51
     > ,5.42,5.06,4.63,4.41,4.20,6.56,7.76,7.61,6.85,6.57
     > ,6.38,6.06,5.84,5.66,5.67,5.69,5.71,5.72,5.73,5.70
     > ,5.60,5.57,5.35,5.22,5.27,5.26,5.22,5.19,5.17,5.16
     > ,5.12,5.09,5.07,5.07,5.06,5.06,5.01,4.92,4.84,4.81
     > ,4.74,4.76,4.77,4.79,4.80,4.79,4.77,4.86,4.96,5.09
     > ,5.12,5.24,5.30,5.46,5.49,5.14,4.92,16.39,49.27,45.07
     > ,27.31,21.62,18.90,15.40,10.40,8.00,6.45,6.33,6.26,6.16
     > ,6.09,5.82,5.55,5.42,5.03,4.89,4.61,4.47,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA OP2P/0.01,0.01,0.01,0.02,0.02,0.02,0.02,0.02,0.02,0.02
     > ,0.02,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03
     > ,0.03,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04
     > ,0.04,0.04,0.04,0.04,0.05,0.05,0.05,0.05,0.05,0.05
     > ,0.05,0.05,0.06,0.05,0.06,0.06,0.06,0.06,0.06,0.07
     > ,0.07,0.07,0.07,0.07,0.07,0.07,0.08,0.08,0.07,0.07
     > ,0.08,0.08,0.08,0.08,0.09,0.09,0.09,0.09,0.09,0.10
     > ,0.09,0.10,0.10,0.10,0.11,0.11,0.11,0.11,0.11,0.11
     > ,0.11,0.12,0.12,0.12,0.12,0.12,0.12,0.12,0.12,0.13
     > ,0.13,0.12,0.12,0.13,0.13,0.13,0.14,0.14,0.14,0.14
     > ,0.14,0.14,0.14,0.15,0.15,0.15,0.15,0.15,0.15,0.16
     > ,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.17
     > ,0.17,0.17,0.17,0.17,0.17,0.17,0.17,0.17,0.18,0.18
     > ,0.18,0.18,0.18,0.18,0.19,0.19,0.19,0.19,0.19,0.19
     > ,0.19,0.20,0.20,0.20,0.20,0.20,0.21,0.21,0.21,0.21
     > ,0.21,0.21,0.21,0.22,0.22,0.22,0.22,0.22,0.22,0.22
     > ,0.23,0.23,0.23,0.23,0.23,0.24,0.24,0.24,0.24,0.25
     > ,0.25,0.25,0.25,0.25,0.25,0.26,0.26,0.26,0.26,0.27
     > ,0.27,0.27,0.28,0.28,0.28,0.28,0.28,0.29,0.30,0.30
     > ,0.30,0.30,0.31,0.31,0.31,0.32,0.33,0.33,0.33,0.34
     > ,0.34,0.34,0.35,0.35,0.37,0.38,0.38,0.39,0.39,0.39
     > ,0.39,0.42,0.42,0.42,0.43,0.43,0.43,0.43,0.43,0.43
     > ,0.42,0.42,0.42,0.42,0.42,0.42,0.42,0.46,0.49,0.51
     > ,0.52,0.57,0.58,0.59,0.62,0.63,0.63,0.64,0.65,0.65
     > ,0.67,0.68,0.70,0.70,0.72,0.72,0.73,0.73,0.75,0.76
     > ,0.76,0.77,0.77,0.78,0.78,0.78,0.79,0.79,0.79,0.80
     > ,0.81,0.81,0.81,0.81,0.81,0.82,0.83,0.83,0.84,0.84
     > ,0.84,0.85,0.86,0.86,0.86,0.87,0.87,0.88,0.89,0.89
     > ,0.90,0.90,0.91,0.92,0.93,0.94,0.94,0.95,0.96,0.96
     > ,0.96,0.97,0.98,0.98,0.98,0.99,1.00,1.00,1.00,1.01
     > ,1.02,1.03,1.03,1.04,1.04,1.05,1.05,1.06,1.07,1.07
     > ,1.08,1.08,1.08,1.09,1.10,1.11,1.11,1.11,1.12,1.12
     > ,1.12,1.13,1.14,1.14,1.15,1.21,1.21,1.22,1.23,1.23
     > ,1.24,1.25,1.26,1.27,1.27,1.27,1.28,1.29,1.30,1.30
     > ,1.31,1.31,1.32,1.32,1.33,1.34,1.34,1.35,1.36,1.36
     > ,1.37,1.38,1.38,1.39,1.39,1.40,1.40,1.40,1.41,1.41
     > ,1.42,1.44,1.45,1.45,1.45,1.49,1.50,1.50,1.51,1.52
     > ,1.53,1.54,1.54,1.54,1.54,1.55,1.55,1.55,1.56,1.58
     > ,1.59,1.59,1.63,1.64,1.65,1.65,1.75,1.77,1.78,1.78
     > ,1.78,1.78,1.79,1.79,1.80,1.80,1.81,1.81,1.82,1.84
     > ,1.85,1.94,2.02,2.03,2.03,2.05,2.06,2.06,2.07,2.11
     > ,2.15,2.19,2.19,2.23,2.27,2.27,2.28,2.29,2.31,2.31
     > ,2.34,2.36,2.39,2.40,2.43,2.43,2.46,2.47,2.49,2.55
     > ,2.63,2.94,2.95,2.93,2.92,2.91,2.88,2.85,2.94,2.92
     > ,2.90,2.88,2.86,2.87,2.87,2.88,2.89,2.90,2.90,2.91
     > ,2.91,2.92,2.92,2.93,2.94,2.94,2.95,2.95,2.96,2.96
     > ,2.97,2.97,2.98,2.98,2.99,2.99,2.99,2.98,2.98,2.97
     > ,2.97,3.08,3.08,3.07,3.09,3.13,3.17,3.22,3.24,3.16
     > ,2.99,2.94,3.48,3.37,3.13,3.10,3.01,3.00,2.86,2.84
     > ,3.78,3.61,3.27,3.16,3.00,3.01,3.04,3.05,3.02,2.99
     > ,2.97,2.91,2.75,2.70,2.97,3.44,4.08,3.74,3.46,3.41
     > ,3.31,3.21,3.19,3.19,3.20,3.20,3.21,3.21,3.22,3.22
     > ,3.23,3.23,3.24,3.24,3.25,3.25,3.25,3.26,3.27,3.27
     > ,3.24,3.23,3.19,3.11,3.09,3.05,3.14,3.28,3.35,3.33
     > ,3.29,4.35,4.20,3.70,3.65,3.63,3.54,3.52,3.51,3.50
     > ,3.48,3.47,3.44,3.42,3.39,3.34,3.29,3.27,3.24,3.28
     > ,3.31,3.31,3.30,3.29,3.27,3.26,3.25,3.24,3.23,3.22
     > ,3.22,3.21,3.18,3.18,3.18,3.18,3.19,3.19,3.20,3.21
     > ,3.23,3.24,3.24,3.24,3.26,3.22,3.18,3.20,3.21,3.22
     > ,3.24,3.25,3.26,3.27,3.27,3.28,3.29,3.30,3.32,3.33
     > ,3.35,3.35,3.36,3.41,3.51,3.47,3.14,3.15,3.18,3.21
     > ,3.32,3.37,3.45,3.51,3.54,3.61,3.71,3.72,3.68,3.64
     > ,3.64,3.63,3.62,3.62,3.61,3.61,3.59,3.59,3.58,3.58
     > ,3.56,3.55,3.55,3.54,3.53,3.53,3.53,3.51,3.51,3.51
     > ,3.50,3.50,3.49,3.48,3.47,3.46,3.43,3.40,3.38,3.36
     > ,3.35,3.33,3.31,3.30,3.27,3.24,3.23,3.22,3.22,3.15
     > ,3.01,2.89,2.81,2.61,2.40,2.32,2.26,2.20,2.04,2.00
     > ,1.94,1.80,1.74,1.60,1.50,1.46,1.40,1.20,1.30,1.43
     > ,1.65,1.69,1.82,1.94,2.19,2.24,2.29,2.43,2.65,2.82
     > ,2.85,2.92,3.04,3.16,3.40,3.50,3.64,3.64,3.64,3.64
     > ,3.64,3.65,3.65,3.65,3.65,3.65,3.65,3.65,3.65,3.65
     > ,3.60,3.47,3.40,3.31,3.11,2.94,2.84,2.78,1.93,0.55
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA OP2E/0.00,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01
     > ,0.01,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02
     > ,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02
     > ,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.03,0.03,0.03
     > ,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03
     > ,0.03,0.03,0.03,0.03,0.03,0.03,0.04,0.03,0.03,0.03
     > ,0.03,0.03,0.04,0.04,0.04,0.04,0.04,0.04,0.04,0.04
     > ,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05
     > ,0.05,0.06,0.06,0.06,0.06,0.06,0.05,0.05,0.06,0.06
     > ,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.07,0.07,0.07
     > ,0.06,0.06,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07
     > ,0.07,0.07,0.08,0.08,0.08,0.07,0.07,0.08,0.08,0.08
     > ,0.08,0.08,0.08,0.07,0.07,0.08,0.08,0.08,0.08,0.08
     > ,0.08,0.08,0.08,0.09,0.09,0.09,0.09,0.08,0.09,0.09
     > ,0.09,0.09,0.09,0.09,0.10,0.10,0.10,0.10,0.09,0.09
     > ,0.09,0.10,0.10,0.10,0.10,0.09,0.11,0.11,0.11,0.11
     > ,0.11,0.10,0.10,0.11,0.11,0.11,0.10,0.11,0.12,0.12
     > ,0.11,0.11,0.12,0.12,0.12,0.12,0.12,0.12,0.13,0.13
     > ,0.13,0.13,0.13,0.13,0.12,0.12,0.14,0.13,0.13,0.14
     > ,0.14,0.13,0.15,0.15,0.13,0.14,0.15,0.16,0.16,0.16
     > ,0.16,0.16,0.17,0.17,0.18,0.18,0.18,0.18,0.18,0.19
     > ,0.19,0.20,0.20,0.20,0.21,0.21,0.20,0.20,0.20,0.20
     > ,0.20,0.20,0.20,0.20,0.20,0.20,0.20,0.22,0.22,0.23
     > ,0.23,0.26,0.26,0.27,0.28,0.29,0.29,0.29,0.30,0.30
     > ,0.30,0.31,0.32,0.32,0.33,0.33,0.33,0.33,0.34,0.34
     > ,0.35,0.35,0.35,0.35,0.35,0.36,0.36,0.36,0.36,0.36
     > ,0.37,0.37,0.37,0.37,0.37,0.37,0.38,0.38,0.38,0.38
     > ,0.38,0.39,0.39,0.39,0.39,0.40,0.40,0.40,0.40,0.40
     > ,0.41,0.41,0.41,0.42,0.42,0.43,0.43,0.43,0.43,0.44
     > ,0.44,0.44,0.44,0.44,0.45,0.45,0.45,0.45,0.45,0.46
     > ,0.46,0.47,0.47,0.47,0.47,0.48,0.48,0.48,0.48,0.49
     > ,0.49,0.49,0.49,0.50,0.50,0.50,0.50,0.50,0.51,0.51
     > ,0.51,0.51,0.52,0.52,0.52,0.52,0.53,0.53,0.53,0.54
     > ,0.54,0.54,0.55,0.55,0.55,0.55,0.56,0.56,0.56,0.57
     > ,0.57,0.57,0.57,0.57,0.58,0.58,0.58,0.59,0.59,0.59
     > ,0.60,0.60,0.60,0.61,0.61,0.61,0.61,0.61,0.61,0.62
     > ,0.62,0.63,0.63,0.63,0.63,0.65,0.65,0.65,0.66,0.66
     > ,0.66,0.67,0.67,0.67,0.67,0.67,0.67,0.68,0.68,0.69
     > ,0.69,0.69,0.71,0.71,0.72,0.72,0.73,0.66,0.71,0.74
     > ,0.74,0.74,0.74,0.69,0.67,0.67,0.68,0.68,0.68,0.69
     > ,0.69,0.75,0.81,0.81,0.81,0.82,0.82,0.82,0.82,0.81
     > ,0.80,0.79,0.79,0.80,0.82,0.82,0.82,0.82,0.83,0.83
     > ,0.84,0.85,0.86,0.86,0.87,0.88,0.79,0.79,0.78,0.78
     > ,0.77,0.83,0.83,0.82,0.82,0.82,0.81,0.80,0.79,0.79
     > ,0.78,0.78,0.77,0.77,0.77,0.78,0.78,0.78,0.78,0.78
     > ,0.78,0.79,0.79,0.79,0.79,0.79,0.79,0.80,0.80,0.80
     > ,0.80,0.80,0.69,0.69,0.69,0.69,0.69,0.69,0.69,0.69
     > ,0.68,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA OP4E/0.00,0.00,0.00,0.00,0.01,0.01,0.01,0.01,0.01,0.01
     > ,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01
     > ,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.02
     > ,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02
     > ,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.03
     > ,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03
     > ,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.04,0.03,0.03
     > ,0.03,0.03,0.03,0.03,0.04,0.04,0.04,0.04,0.04,0.04
     > ,0.04,0.04,0.04,0.04,0.04,0.04,0.05,0.05,0.05,0.05
     > ,0.05,0.05,0.05,0.04,0.05,0.05,0.05,0.05,0.05,0.05
     > ,0.05,0.05,0.05,0.06,0.06,0.06,0.06,0.06,0.06,0.05
     > ,0.05,0.05,0.05,0.06,0.06,0.06,0.06,0.06,0.06,0.06
     > ,0.06,0.06,0.06,0.06,0.06,0.06,0.07,0.07,0.07,0.07
     > ,0.06,0.06,0.07,0.07,0.07,0.07,0.07,0.06,0.06,0.06
     > ,0.07,0.07,0.08,0.08,0.08,0.07,0.07,0.08,0.08,0.08
     > ,0.08,0.08,0.08,0.07,0.07,0.08,0.08,0.08,0.08,0.09
     > ,0.09,0.08,0.08,0.09,0.09,0.09,0.09,0.08,0.09,0.09
     > ,0.09,0.09,0.09,0.10,0.08,0.10,0.10,0.10,0.09,0.10
     > ,0.10,0.10,0.11,0.11,0.11,0.11,0.09,0.11,0.11,0.11
     > ,0.11,0.10,0.12,0.12,0.12,0.12,0.12,0.13,0.13,0.13
     > ,0.13,0.13,0.13,0.13,0.14,0.15,0.15,0.15,0.15,0.15
     > ,0.15,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.16
     > ,0.16,0.16,0.16,0.16,0.16,0.16,0.16,0.17,0.18,0.19
     > ,0.19,0.21,0.21,0.21,0.22,0.23,0.23,0.23,0.24,0.24
     > ,0.24,0.25,0.25,0.25,0.26,0.26,0.26,0.27,0.27,0.27
     > ,0.28,0.28,0.28,0.28,0.28,0.28,0.29,0.29,0.29,0.29
     > ,0.29,0.29,0.29,0.30,0.30,0.30,0.30,0.30,0.30,0.30
     > ,0.31,0.31,0.31,0.31,0.31,0.32,0.32,0.32,0.32,0.32
     > ,0.33,0.33,0.33,0.33,0.34,0.34,0.34,0.34,0.35,0.35
     > ,0.35,0.35,0.35,0.36,0.36,0.36,0.36,0.36,0.36,0.37
     > ,0.32,0.36,0.38,0.38,0.38,0.33,0.33,0.34,0.34,0.34
     > ,0.34,0.34,0.35,0.35,0.35,0.35,0.35,0.35,0.36,0.36
     > ,0.36,0.36,0.36,0.36,0.36,0.37,0.37,0.37,0.37,0.37
     > ,0.38,0.38,0.38,0.39,0.39,0.39,0.39,0.39,0.39,0.40
     > ,0.40,0.40,0.40,0.40,0.40,0.41,0.41,0.41,0.41,0.41
     > ,0.42,0.42,0.42,0.42,0.42,0.39,0.37,0.37,0.37,0.37
     > ,0.37,0.38,0.38,0.38,0.38,0.39,0.39,0.39,0.39,0.40
     > ,0.40,0.40,0.40,0.40,0.40,0.40,0.40,0.41,0.41,0.37
     > ,0.36,0.35,0.35,0.36,0.36,0.36,0.37,0.37,0.37,0.37
     > ,0.37,0.37,0.37,0.37,0.37,0.37,0.30,0.30,0.30,0.31
     > ,0.31,0.14,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA OP/0.04,0.06,0.07,0.07,0.07,0.07,0.08,0.08,0.08,0.10
     > ,0.11,0.14,0.14,0.16,0.16,0.16,0.18,0.18,0.18,0.18
     > ,0.18,0.18,0.18,0.18,0.19,0.20,0.20,0.21,0.21,0.22
     > ,0.22,0.23,0.24,0.24,0.25,0.25,0.26,0.26,0.27,0.28
     > ,0.28,0.29,0.29,0.30,0.31,0.32,0.34,0.35,0.35,0.36
     > ,0.37,0.38,0.38,0.39,0.39,0.39,0.40,0.41,0.41,0.42
     > ,0.43,0.44,0.44,0.45,0.47,0.48,0.48,0.50,0.51,0.52
     > ,0.53,0.54,0.55,0.55,0.57,0.58,0.58,0.59,0.59,0.61
     > ,0.62,0.63,0.64,0.64,0.65,0.65,0.66,0.66,0.67,0.68
     > ,0.68,0.69,0.70,0.70,0.72,0.73,0.74,0.75,0.76,0.76
     > ,0.77,0.78,0.79,0.79,0.81,0.82,0.82,0.83,0.84,0.85
     > ,0.85,0.86,0.86,0.87,0.87,0.88,0.88,0.89,0.89,0.90
     > ,0.90,0.91,0.92,0.92,0.93,0.93,0.94,0.95,0.96,0.96
     > ,0.97,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.04,1.05
     > ,1.06,1.07,1.07,1.10,1.11,1.11,1.12,1.13,1.14,1.15
     > ,1.15,1.16,1.17,1.18,1.18,1.19,1.20,1.21,1.21,1.22
     > ,1.23,1.24,1.25,1.25,1.27,1.28,1.29,1.31,1.33,1.33
     > ,1.33,1.34,1.35,1.36,1.37,1.38,1.41,1.42,1.43,1.45
     > ,1.46,1.47,1.49,1.51,1.52,1.53,1.54,1.58,1.60,1.61
     > ,1.61,1.64,1.65,1.68,1.69,1.71,1.76,1.77,1.79,1.80
     > ,1.82,1.84,1.86,1.89,1.97,2.04,2.04,2.05,2.06,2.07
     > ,2.08,2.20,2.23,2.24,2.28,2.27,2.26,2.24,2.24,2.23
     > ,2.22,2.21,2.21,2.19,2.19,2.18,2.18,2.36,2.41,2.51
     > ,2.54,2.80,2.85,2.87,3.02,3.07,3.09,3.12,3.17,3.19
     > ,3.26,3.33,3.38,3.39,3.49,3.52,3.53,3.55,3.66,3.66
     > ,3.69,3.73,3.74,3.75,3.77,3.78,3.79,3.83,3.83,3.87
     > ,3.88,3.89,3.90,3.91,3.92,3.93,3.97,3.98,4.02,4.03
     > ,4.04,4.09,4.11,4.12,4.13,4.16,4.18,4.20,4.25,4.25
     > ,4.30,4.30,4.33,4.37,4.41,4.45,4.48,4.50,4.54,4.55
     > ,4.57,4.62,4.62,4.63,4.65,4.68,4.72,4.72,4.72,4.76
     > ,4.79,4.84,4.86,4.87,4.88,4.93,4.93,4.96,4.99,5.01
     > ,5.05,5.05,5.07,5.11,5.12,5.15,5.16,5.16,5.21,5.22
     > ,5.23,5.26,5.29,5.30,5.32,5.34,5.37,5.40,5.42,5.44
     > ,5.45,5.50,5.53,5.57,5.57,5.59,5.62,5.66,5.68,5.71
     > ,5.73,5.74,5.76,5.77,5.79,5.81,5.84,5.88,5.91,5.92
     > ,5.98,6.00,6.02,6.05,6.06,6.07,6.08,6.10,6.14,6.15
     > ,6.19,6.25,6.30,6.32,6.33,6.48,6.50,6.52,6.57,6.59
     > ,6.65,6.68,6.69,6.70,6.72,6.74,6.74,6.75,6.79,6.85
     > ,6.90,6.93,7.08,7.13,7.16,7.17,7.31,7.35,7.40,7.41
     > ,7.42,7.43,7.44,7.48,7.48,7.49,7.54,7.55,7.58,7.68
     > ,7.70,7.90,8.07,8.11,8.12,8.22,8.24,8.25,8.27,8.43
     > ,8.60,8.76,8.77,8.92,9.09,9.09,9.11,9.16,9.25,9.25
     > ,9.37,9.44,9.57,9.60,9.72,9.73,9.84,9.90,9.95,10.20
     > ,10.50,11.78,11.80,11.71,11.66,11.64,11.54,11.40,11.32,11.24
     > ,11.16,11.08,11.00,11.03,11.06,11.09,11.11,11.14,11.17,11.20
     > ,11.21,11.23,11.24,11.27,11.30,11.32,11.34,11.36,11.38,11.40
     > ,11.42,11.44,11.46,11.48,11.50,11.49,11.48,11.46,11.45,11.43
     > ,11.41,11.39,11.39,11.38,11.43,11.60,11.76,11.92,12.00,11.69
     > ,11.08,10.90,12.90,12.47,11.60,11.47,11.13,11.10,10.61,10.50
     > ,14.00,13.37,12.10,11.70,11.10,11.15,11.27,11.30,11.20,11.08
     > ,11.00,10.76,10.18,10.00,11.00,12.76,15.10,13.85,12.80,12.61
     > ,12.24,11.87,11.80,11.82,11.84,11.86,11.88,11.90,11.92,11.93
     > ,11.95,11.97,11.99,12.01,12.03,12.03,12.05,12.07,12.10,12.10
     > ,12.00,11.97,11.80,11.50,11.46,11.30,11.61,12.14,12.40,12.33
     > ,12.20,16.10,15.57,13.70,13.50,13.43,13.10,13.04,13.00,12.95
     > ,12.90,12.86,12.73,12.65,12.57,12.38,12.19,12.10,12.00,12.16
     > ,12.27,12.26,12.21,12.17,12.12,12.08,12.03,11.99,11.94,11.93
     > ,11.93,11.90,11.77,11.76,11.77,11.79,11.81,11.82,11.87,11.90
     > ,11.96,12.00,12.01,12.01,12.06,12.10,12.24,12.30,12.36,12.39
     > ,12.48,12.50,12.53,12.59,12.59,12.61,12.67,12.70,12.76,12.80
     > ,12.87,12.90,12.92,12.95,13.00,13.01,13.08,13.09,13.10,13.12
     > ,13.18,13.20,13.22,13.24,13.25,13.27,13.30,13.30,13.34,13.38
     > ,13.38,13.39,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40
     > ,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40
     > ,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40
     > ,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40,13.40
     > ,13.39,13.38,13.38,13.37,13.36,13.36,13.35,13.35,13.34,13.34
     > ,13.34,13.33,13.33,13.32,13.32,13.31,13.31,13.30,13.29,13.27
     > ,13.25,13.24,13.23,13.21,13.18,13.17,13.17,13.15,13.12,13.10
     > ,13.10,13.09,13.07,13.06,13.03,13.02,13.00,12.99,12.96,12.93
     > ,12.92,12.88,12.84,12.80,12.76,12.72,12.71,12.68,12.64,12.60
     > ,12.57,12.49,12.44,12.38,12.26,12.15,12.08,12.05,12.00,11.92
     > ,11.89,11.72,11.67,11.56,11.39,11.30,11.23,11.16,11.07,10.98
     > ,10.92,10.75,10.64,10.61,10.57,10.51,10.41,10.38,10.30,10.25
     > ,10.21,10.15,10.08,10.00,10.30,10.60,12.00,17.67,15.40,14.52
     > ,14.15,13.90,14.29,14.68,15.07,15.20,14.41,14.02,12.84,11.66
     > ,9.69,9.65,9.52,9.29,9.25,9.70,10.20,11.20,13.13,24.68
     > ,25.49,22.56,19.54,16.51,15.50,13.35,11.20,12.48,12.41,11.92
     > ,11.43,9.80,9.47,9.48,9.48,9.49,9.49,9.51,9.51,9.57
     > ,9.51,9.37,9.00,9.01,9.02,9.04,9.06,8.91,8.80,9.36
     > ,9.70,40.00,37.09,32.71,28.34,19.60,16.17,14.79,14.11,12.05
     > ,9.99,9.30,8.45,8.03,7.60,11.80,13.90,13.59,12.02,11.46
     > ,11.08,10.46,9.99,9.60,9.64,9.71,9.78,9.82,9.88,9.85
     > ,9.74,9.71,9.36,9.15,9.30,9.30,9.27,9.26,9.25,9.24
     > ,9.21,9.19,9.20,9.21,9.24,9.24,9.17,9.05,8.93,8.88
     > ,8.79,8.84,8.91,8.97,9.00,9.00,9.00,8.96,8.91,8.94
     > ,8.94,8.97,8.99,9.02,8.90,8.75,8.60,26.10,61.50,54.30
     > ,33.90,27.10,24.07,20.02,13.96,10.92,8.90,8.99,9.04,9.12
     > ,9.17,8.98,8.78,8.69,8.40,8.33,8.19,8.12,8.08,8.03
     > ,6.97,5.21,4.15,8.62,11.81,15.00,15.34,16.01,16.69,10.40
     > ,12.30,15.15,12.50,8.90,14.37,16.20,9.08,7.30,18.41,18.91
     > ,11.77,9.98,8.20,6.33,5.71,10.08,14.75,21.76,17.10,10.10
     > ,5.42,4.85,4.70,5.49,6.80,20.10,33.40,20.33,15.10,9.82
     > ,5.20,5.00,4.61,4.35,4.15,4.36,4.76,4.90,22.34,28.16
     > ,36.88,45.60,40.19,34.77,26.65,13.11,7.70,9.57,10.83,12.70
     > ,11.38,10.94,9.71,8.30,6.54,3.90,4.05,4.06,4.05,4.05
     > ,4.04,4.04,4.04,4.03,4.03,4.02,4.02,4.02,4.01,4.01
     > ,4.00,4.00,3.99,3.99,3.99,3.98,3.97,3.97,3.96,3.95
     > ,3.95,3.95,3.94,3.93,3.93,3.93,3.92,3.92,3.91,3.90
     > ,3.90,3.89,3.88,3.88,3.87,3.86,3.85,3.84,3.83,3.82
     > ,3.81,3.80,3.79,3.79,3.78,3.77,3.76,3.75,3.74,3.73
     > ,3.73,3.72,3.71,3.70,3.70,3.70,3.70,3.70,3.70,3.70
     > ,3.70,3.70,3.70,3.70,3.70,3.70,3.70,3.70,3.70,3.70
     > ,3.71,3.71,3.72,3.72,3.72,3.73,3.73,3.74,3.74,3.75
     > ,3.76,3.76,3.77,3.77,3.77,3.78,3.78,4.23,3.78,4.86
     > ,19.97,2.70,2.74,2.80,2.86,2.92,2.94,3.03,3.15,3.23
     > ,3.26,3.38,3.44,3.50,3.55,3.61,3.67,3.73,3.80,3.84
     > ,3.90,3.96,36.41,9.14,5.34,48.94,7.20,11.73,3.64,15.37
     > ,3.64,5.91,1.62,2.06,2.13,2.14,2.17,2.19,2.20,2.22
     > ,2.24,2.27,2.31,15.23,2.82,32.62,3.39,7.90,4.74,2.80
     > ,3.67,3.95,1.97,2.21,2.38,2.79,2.95,3.19,3.60,3.93
     > ,4.42,4.83,5.17,65.14,4.14,1.86,3.10,22.02,2.59,7.40
     > ,1.96,1.55,1.68,1.82,1.90,1.95,2.09,2.22,2.30,2.36
     > ,2.49,2.76,2.82,3.03,3.17,3.27,3.30,3.30,3.30,3.30
     > ,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30
     > ,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30
     > ,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30,3.30
     > ,3.30,3.30,3.30,3.30,3.29,3.29,3.29,3.28,3.28,3.27
     > ,3.27,3.26,3.26,3.26,3.25,3.25,3.24,3.23,3.22,3.22
     > ,3.22,3.21,3.21,3.21,3.20,3.19,3.18,3.17,3.16,3.16
     > ,3.16,3.15,3.14,3.14,3.13,3.12,3.11,3.10,3.10,3.10
     > ,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10
     > ,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10
     > ,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10,3.10
     > ,3.77,3.94,4.27,4.77,5.19,5.61,6.02,6.44,6.86,7.28
     > ,7.44,7.70,8.11,8.28,8.53,8.95,9.37,11.82,34.88,9.93
     > ,4.73,6.81,5.20,5.11,4.88,4.66,4.43,4.21,3.98,3.89
     > ,3.75,3.62,3.53,3.30,3.08,2.85,2.84,2.83,2.82,2.81
     > ,2.79,2.78,2.76,2.75,2.73,2.72,2.70,2.71,2.71,2.72
     > ,2.72,2.73,2.73,2.74,2.74,2.75,2.75,2.75,2.76,2.77
     > ,2.77,2.78,2.79,2.79,2.80,2.81,2.82,2.82,2.83,2.84
     > ,2.85,2.85,2.84,2.83,2.82,2.82,2.81,2.79,2.78,2.77
     > ,2.77,2.76,2.75,2.75,2.75,2.75,2.75,2.75,2.68,2.62
     > ,2.55,2.49,2.45,1.15,0.75,0.49,0.30,0.20,0.14,0.06
     > ,0.04,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA N2_ABS/0.80,0.96,0.97,0.99,1.01,1.02,0.91,0.51,0.08,0.10
     > ,0.11,0.15,0.15,0.17,0.17,0.17,0.18,0.18,0.19,0.19
     > ,0.20,0.20,0.21,0.21,0.23,0.23,0.24,0.24,0.24,0.25
     > ,0.26,0.27,0.28,0.28,0.29,0.29,0.30,0.30,0.31,0.31
     > ,0.32,0.33,0.33,0.34,0.35,0.36,0.37,0.38,0.38,0.39
     > ,0.39,0.40,0.40,0.41,0.41,0.41,0.42,0.42,0.43,0.43
     > ,0.44,0.44,0.45,0.45,0.46,0.46,0.47,0.48,0.50,0.51
     > ,0.52,0.53,0.53,0.54,0.56,0.56,0.57,0.57,0.57,0.59
     > ,0.60,0.60,0.61,0.62,0.62,0.62,0.63,0.63,0.64,0.65
     > ,0.65,0.66,0.66,0.66,0.68,0.68,0.69,0.69,0.70,0.70
     > ,0.71,0.71,0.72,0.73,0.73,0.74,0.74,0.75,0.76,0.78
     > ,0.78,0.79,0.80,0.81,0.82,0.82,0.83,0.84,0.85,0.85
     > ,0.86,0.87,0.87,0.89,0.89,0.90,0.91,0.93,0.94,0.94
     > ,0.95,0.96,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.04
     > ,1.05,1.06,1.07,1.10,1.11,1.12,1.12,1.13,1.15,1.16
     > ,1.16,1.17,1.19,1.20,1.20,1.22,1.23,1.24,1.24,1.25
     > ,1.26,1.28,1.28,1.29,1.31,1.32,1.34,1.36,1.39,1.39
     > ,1.39,1.40,1.41,1.42,1.44,1.45,1.49,1.50,1.51,1.53
     > ,1.54,1.55,1.59,1.60,1.62,1.62,1.64,1.68,1.72,1.73
     > ,1.73,1.77,1.78,1.82,1.84,1.87,1.95,1.97,2.00,2.02
     > ,2.06,2.09,2.11,2.15,2.27,2.36,2.37,2.38,2.39,2.41
     > ,2.42,2.56,2.61,2.62,2.68,2.77,2.82,2.95,2.98,3.02
     > ,3.14,3.21,3.22,3.32,3.35,3.42,3.46,3.60,3.63,3.71
     > ,3.73,3.92,3.97,3.99,4.11,4.14,4.16,4.18,4.22,4.24
     > ,4.30,4.35,4.40,4.41,4.48,4.51,4.52,4.54,4.64,4.64
     > ,4.69,4.77,4.79,4.81,4.84,4.86,4.89,4.96,4.98,5.05
     > ,5.08,5.10,5.13,5.14,5.16,5.17,5.27,5.29,5.36,5.38
     > ,5.41,5.50,5.55,5.57,5.59,5.66,5.68,5.73,5.83,5.84
     > ,5.94,5.95,6.01,6.09,6.20,6.28,6.36,6.40,6.49,6.52
     > ,6.57,6.68,6.69,6.71,6.78,6.85,6.96,6.97,6.98,7.10
     > ,7.17,7.30,7.38,7.40,7.41,7.57,7.58,7.79,7.77,7.85
     > ,7.96,7.97,8.01,8.13,8.17,8.26,8.28,8.29,8.45,8.47
     > ,8.49,8.60,8.68,8.71,8.76,8.84,8.92,9.02,9.07,9.12
     > ,9.16,9.30,9.38,9.49,9.50,9.54,9.60,9.68,9.73,9.80
     > ,9.84,9.85,9.88,9.90,9.93,9.98,10.02,10.06,10.08,10.09
     > ,10.14,10.16,10.18,10.21,10.21,10.22,10.23,10.25,10.27,10.28
     > ,10.31,10.35,10.38,10.40,10.40,10.49,10.50,10.51,10.54,10.56
     > ,10.59,10.61,10.62,10.63,10.65,10.67,10.68,10.69,10.73,10.78
     > ,10.81,10.82,10.90,10.92,10.95,10.95,11.04,11.07,11.10,11.12
     > ,11.16,11.17,11.19,11.27,11.30,11.32,11.48,11.50,11.54,11.67
     > ,11.70,12.02,12.37,12.45,12.46,12.66,12.72,12.73,12.77,13.13
     > ,13.55,13.95,13.98,14.38,14.82,14.83,14.89,15.04,15.27,15.28
     > ,15.63,15.82,16.18,16.25,16.58,16.61,16.91,17.08,17.53,18.03
     > ,19.05,20.05,20.07,20.18,20.23,20.25,20.38,20.55,20.64,20.74
     > ,20.83,20.93,21.02,21.10,21.19,21.27,21.35,21.44,21.52,21.60
     > ,21.62,21.66,21.68,21.77,21.85,21.93,22.01,22.09,22.17,22.25
     > ,22.32,22.40,22.48,22.56,22.64,22.66,22.69,22.73,22.78,22.82
     > ,22.87,22.92,22.92,22.95,22.96,23.01,23.05,23.10,23.10,23.10
     > ,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10
     > ,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.09
     > ,23.09,23.09,23.08,23.08,23.08,23.08,23.07,23.07,23.07,23.07
     > ,23.06,23.06,23.05,23.05,23.07,23.09,23.12,23.14,23.16,23.16
     > ,23.18,23.20,23.23,23.25,23.27,23.27,23.30,23.34,23.37,23.38
     > ,23.40,23.40,23.44,23.46,23.47,23.48,23.50,23.53,23.55,23.57
     > ,23.60,23.59,23.59,23.58,23.58,23.58,23.57,23.57,23.57,23.56
     > ,23.55,23.55,23.54,23.53,23.53,23.52,23.51,23.50,23.50,23.50
     > ,23.50,23.50,23.50,23.50,23.50,23.50,23.50,23.50,23.50,23.50
     > ,23.50,23.50,23.52,23.52,23.53,23.55,23.56,23.58,23.73,23.83
     > ,24.25,24.58,24.61,24.63,24.86,25.07,25.23,25.30,25.13,25.02
     > ,24.77,24.70,24.52,24.13,24.11,23.97,23.58,23.40,23.15,22.95
     > ,22.64,22.50,22.48,22.45,22.40,22.40,22.40,22.40,22.40,22.40
     > ,22.40,22.40,22.44,22.47,22.49,22.52,22.57,22.58,22.67,22.76
     > ,22.76,22.78,22.79,22.80,22.82,22.83,22.86,22.87,22.88,22.89
     > ,22.92,22.95,22.96,22.98,22.99,23.00,23.01,23.03,23.04,23.05
     > ,23.06,23.07,23.09,23.10,23.11,23.13,23.15,23.18,23.21,23.23
     > ,23.24,23.27,23.28,23.30,23.32,23.35,23.37,23.37,23.38,23.39
     > ,23.41,23.42,23.44,23.46,23.49,23.50,23.51,23.52,23.54,23.55
     > ,23.56,23.58,23.58,23.60,23.62,23.62,23.63,23.66,23.67,23.69
     > ,23.71,23.72,23.73,23.75,23.78,23.78,23.79,23.81,23.83,23.85
     > ,23.85,23.86,23.88,23.89,23.92,23.93,23.95,23.96,23.98,24.00
     > ,24.00,24.03,24.05,24.08,24.10,24.13,24.13,24.15,24.18,24.20
     > ,24.30,25.22,25.74,26.40,26.69,26.94,27.10,25.60,27.47,27.42
     > ,27.40,30.00,40.40,22.40,48.40,32.32,21.60,22.00,28.71,34.30
     > ,28.40,23.00,23.00,23.00,30.41,45.24,67.49,74.90,65.77,60.30
     > ,56.65,49.35,42.05,32.93,25.62,25.42,28.13,29.21,30.29,34.08
     > ,35.70,34.67,33.14,31.60,25.30,32.80,47.80,55.30,77.80,65.57
     > ,45.18,37.03,24.80,24.43,24.36,24.07,23.92,23.70,23.70,23.70
     > ,23.70,23.70,23.70,23.69,23.68,23.67,23.65,23.62,23.60,23.60
     > ,23.60,23.33,23.16,23.11,23.05,23.00,23.00,23.80,24.07,25.13
     > ,27.91,30.69,38.10,39.96,42.74,48.30,76.10,61.30,52.05,42.80
     > ,37.25,33.55,29.85,24.30,23.83,22.89,22.10,22.10,22.10,22.10
     > ,22.10,22.59,24.53,25.50,26.42,28.26,29.18,30.10,28.30,26.54
     > ,25.36,23.40,25.80,25.79,25.79,25.78,25.78,25.77,25.77,25.76
     > ,25.75,25.75,25.74,25.74,25.73,25.73,25.72,25.71,25.71,25.71
     > ,25.70,25.70,20.57,18.55,14.50,14.94,16.26,18.46,20.66,21.54
     > ,23.30,25.06,23.20,27.40,27.05,26.71,25.66,24.97,24.10,26.80
     > ,27.48,30.19,31.54,34.92,37.62,69.30,59.51,53.63,41.88,39.92
     > ,30.60,30.06,29.24,28.16,26.53,25.71,25.17,24.17,23.63,22.90
     > ,22.94,23.00,23.05,23.08,23.16,23.20,22.68,22.42,22.24,22.07
     > ,21.81,21.37,21.11,20.50,30.50,25.00,23.65,20.95,31.80,26.40
     > ,24.04,20.50,18.20,21.84,27.30,26.56,23.60,22.34,17.30,17.48
     > ,18.20,17.20,16.20,13.20,17.92,25.00,27.72,31.80,46.40,34.72
     > ,15.90,18.62,20.66,22.70,51.12,38.60,28.60,18.60,21.52,33.20
     > ,25.85,22.70,20.00,18.20,16.82,14.06,27.30,61.94,30.00,44.00
     > ,25.00,22.00,20.00,17.00,12.50,50.00,69.20,42.68,25.00,16.80
     > ,66.70,62.85,27.86,19.04,11.90,23.82,27.80,27.80,36.90,46.00
     > ,48.00,36.40,25.00,17.30,16.00,25.00,25.00,68.10,18.20,13.30
     > ,18.00,150.00,91.00,70.59,31.80,14.50,13.95,13.74,13.41,13.30
     > ,13.30,13.30,40.90,22.30,27.30,42.30,25.00,32.70,42.30,26.85
     > ,20.67,14.49,19.56,25.00,141.80,50.00,30.00,21.50,18.00,14.50
     > ,33.10,40.90,31.40,143.81,125.00,59.10,24.50,20.20,15.90,38.61
     > ,47.70,40.54,28.60,14.90,14.70,25.15,32.25,50.00,118.20,75.00
     > ,34.00,15.00,17.56,70.92,150.00,234.10,224.36,204.87,155.00,125.00
     > ,50.00,45.50,56.80,35.60,10.50,8.48,7.67,9.04,11.32,13.60
     > ,45.50,25.58,18.60,125.00,58.10,53.24,51.62,125.00,71.80,43.60
     > ,35.14,25.69,15.90,25.00,39.75,54.50,46.22,13.10,8.00,8.00
     > ,8.00,8.00,8.00,8.00,16.70,40.00,104.00,40.00,16.60,1.00
     > ,3.85,6.70,7.82,8.30,8.80,9.96,10.90,12.14,12.75,56.86
     > ,18.00,95.95,159.70,46.70,6.70,4.42,1.00,33.30,33.30,13.92
     > ,1.00,8.00,4.50,3.35,2.05,1.00,1.96,2.81,3.26,3.96
     > ,4.70,15.50,24.70,100.00,33.40,50.00,62.00,80.00,14.70,11.50
     > ,6.70,14.00,24.92,33.92,40.00,44.94,53.00,60.20,66.00,71.20
     > ,74.84,79.00,92.00,12.00,6.60,3.00,7.30,26.70,18.66,13.30
     > ,3.00,3.00,3.00,3.00,8.00,12.24,8.92,6.00,33.30,25.30
     > ,18.10,13.30,3.00,3.00,3.00,3.00,3.00,3.00,3.00,3.00
     > ,3.00,3.00,3.00,3.00,3.00,3.00,5.30,8.50,13.30,13.30
     > ,13.30,83.30,180.00,135.75,3.00,3.00,3.00,20.00,15.98,13.30
     > ,41.32,55.33,3.00,1.00,50.00,26.70,20.00,96.70,13.30,6.70
     > ,3.00,1.80,1.00,26.70,13.30,3.00,1.40,1.00,1.00,1.00
     > ,1.00,1.00,7.28,16.70,46.70,26.66,13.30,6.70,4.48,3.00
     > ,2.24,1.00,3.00,36.48,86.70,66.70,46.70,20.00,13.30,3.00
     > ,173.30,20.00,8.70,12.13,16.70,13.30,6.70,3.00,3.00,3.00
     > ,1.00,6.00,1.00,1.00,1.00,33.40,6.00,3.00,1.00,86.70
     > ,106.70,3.00,1.00,65.00,32.00,14.00,11.00,10.00,6.00,3.00
     > ,120.00,25.00,26.00,80.00,33.00,12.00,10.00,6.00,3.00,3.00
     > ,3.00,3.00,107.00,60.00,54.00,30.00,24.00,23.12,20.00,13.28
     > ,9.60,6.00,4.20,3.00,5.00,110.00,67.00,51.00,1.00,54.00
     > ,27.00,16.80,10.00,1.00,3.00,1.00,3.00,6.60,9.00,15.00
     > ,80.00,18.00,30.00,12.00,20.00,14.00,12.00,3.00,67.00,25.00
     > ,15.00,15.00,1.00,14.00,10.00,3.00,3.00,3.00,3.00,3.00
     > ,3.00,25.00,11.00,104.00,64.00,1.00,3.00,4.20,6.00,10.00
     > ,12.00,11.20,10.00,3.00,31.00,27.00,6.00,10.00,1.00,5.20
     > ,8.00,14.00,12.00,10.00,3.00,1.40,1.00,1.00,15.00,15.00
     > ,9.00,6.00,4.20,3.00,1.00,14.00,6.00,3.40,1.00,1.00
     > ,1.00,82.00,53.80,35.00,15.00,3.00,3.00,3.00,1.00,1.00
     > ,42.00,1.00,1.80,3.00,6.00,3.00,3.00,1.00,97.00,50.75
     > ,27.64,17.00,11.00,3.00,58.00,17.00,18.50,19.10,20.00,15.20
     > ,12.00,22.00,3.00,14.00,9.60,5.20,18.00,13.20,10.00,2.00
     > ,2.00,29.00,3.00,4.50,6.00,6.00,15.00,8.20,7.00,4.72
     > ,4.00,10.00,11.30,5.00,12.00,7.00,6.00,6.00,1.00,7.00
     > ,20.50,25.00,1.00,148.00,106.00,78.00,27.00,29.00,18.00,15.00
     > ,22.00,1.00,27.00,22.80,20.00,11.00,11.00,1.00,1.00,3.00
     > ,3.00,3.00,1.00,1.00,25.00,13.48,1.00,17.20,28.00,2.00
     > ,2.00,12.00,3.00,6.00,18.60,12.00,8.40,3.00,6.00,18.00
     > ,134.00,53.00,62.00,47.00,21.00,1.00,98.00,51.00,30.00,20.40
     > ,10.00,5.80,3.00,3.00,3.00,1.00,67.00,60.00,31.00,20.00
     > ,27.00,10.00,3.00,6.00,2.00,1.00,6.00,27.00,120.00,94.40
     > ,31.00,20.00,3.00,5.00,3.80,3.00,3.00,1.00,2.00,8.00
     > ,5.00,2.00,88.00,33.00,27.00,22.00,26.00,11.00,14.00,25.00
     > ,23.00,18.20,15.00,17.00,15.00,15.00,13.00,10.00,22.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA N2P/0.80,0.96,0.97,0.99,0.61,0.61,0.55,0.31,0.05,0.06
     > ,0.06,0.09,0.09,0.10,0.10,0.10,0.11,0.11,0.11,0.11
     > ,0.12,0.12,0.13,0.13,0.14,0.14,0.14,0.14,0.15,0.15
     > ,0.16,0.16,0.16,0.17,0.17,0.18,0.18,0.18,0.19,0.19
     > ,0.19,0.20,0.20,0.20,0.21,0.22,0.22,0.23,0.23,0.24
     > ,0.24,0.24,0.25,0.25,0.25,0.25,0.25,0.26,0.26,0.26
     > ,0.27,0.27,0.27,0.27,0.28,0.28,0.29,0.29,0.30,0.31
     > ,0.32,0.32,0.33,0.33,0.34,0.34,0.35,0.35,0.35,0.36
     > ,0.37,0.37,0.37,0.38,0.38,0.38,0.38,0.39,0.39,0.40
     > ,0.40,0.40,0.40,0.41,0.42,0.42,0.42,0.43,0.43,0.43
     > ,0.43,0.44,0.44,0.45,0.45,0.45,0.46,0.46,0.47,0.48
     > ,0.48,0.49,0.49,0.50,0.50,0.51,0.51,0.52,0.52,0.53
     > ,0.53,0.54,0.54,0.55,0.55,0.56,0.56,0.57,0.58,0.58
     > ,0.59,0.59,0.60,0.61,0.62,0.62,0.63,0.64,0.64,0.64
     > ,0.65,0.66,0.66,0.68,0.69,0.69,0.69,0.70,0.71,0.72
     > ,0.72,0.73,0.74,0.74,0.75,0.76,0.76,0.77,0.77,0.78
     > ,0.78,0.79,0.80,0.80,0.82,0.82,0.84,0.85,0.86,0.87
     > ,0.87,0.88,0.88,0.89,0.90,0.91,0.93,0.93,0.94,0.96
     > ,0.96,0.97,0.99,1.00,1.01,1.02,1.03,1.06,1.08,1.08
     > ,1.09,1.11,1.12,1.15,1.17,1.21,1.28,1.29,1.31,1.33
     > ,1.35,1.37,1.38,1.41,1.48,1.54,1.54,1.55,1.55,1.57
     > ,1.58,1.69,1.73,1.73,1.77,1.84,1.88,1.97,2.00,2.03
     > ,2.09,2.14,2.14,2.20,2.22,2.26,2.28,2.35,2.37,2.41
     > ,2.42,2.54,2.57,2.58,2.66,2.68,2.69,2.70,2.73,2.74
     > ,2.78,2.81,2.84,2.84,2.89,2.91,2.92,2.93,2.98,2.99
     > ,3.01,3.06,3.07,3.08,3.09,3.11,3.12,3.16,3.17,3.20
     > ,3.22,3.23,3.24,3.25,3.26,3.27,3.32,3.33,3.37,3.38
     > ,3.40,3.45,3.48,3.49,3.50,3.54,3.56,3.59,3.65,3.65
     > ,3.72,3.72,3.76,3.80,3.87,3.92,3.96,3.99,4.04,4.07
     > ,4.09,4.16,4.17,4.18,4.22,4.27,4.33,4.34,4.35,4.41
     > ,4.45,4.52,4.57,4.58,4.59,4.68,4.68,4.75,4.79,4.84
     > ,4.91,4.91,4.93,5.01,5.03,5.09,5.10,5.11,5.20,5.21
     > ,5.23,5.29,5.34,5.36,5.40,5.45,5.51,5.57,5.61,5.64
     > ,5.67,5.77,5.82,5.90,5.91,5.95,6.00,6.07,6.11,6.17
     > ,6.22,6.24,6.28,6.29,6.34,6.40,6.45,6.51,6.55,6.56
     > ,6.65,6.68,6.71,6.78,6.80,6.83,6.84,6.89,6.96,6.99
     > ,7.03,7.12,7.17,7.19,7.20,7.42,7.44,7.46,7.54,7.57
     > ,7.64,7.68,7.69,7.71,7.73,7.75,7.76,7.77,7.82,7.88
     > ,7.93,7.95,8.10,8.14,8.18,8.18,8.31,8.36,8.40,8.43
     > ,8.48,8.50,8.54,8.65,8.68,8.71,8.88,8.91,9.04,9.18
     > ,9.21,9.57,9.92,10.00,10.01,10.22,10.29,10.30,10.34,10.68
     > ,11.11,11.60,11.65,12.20,12.70,12.72,12.79,12.99,13.28,13.30
     > ,13.72,13.96,14.40,14.49,14.92,14.96,15.35,15.58,16.15,16.80
     > ,17.90,18.98,19.00,19.11,19.17,19.19,19.33,19.50,19.60,19.70
     > ,19.80,19.90,20.00,20.10,20.20,20.30,20.40,20.50,20.60,20.70
     > ,20.72,20.77,20.80,20.90,21.00,21.07,21.14,21.22,21.29,21.36
     > ,21.43,21.50,21.58,21.65,21.72,21.74,21.76,21.81,21.85,21.89
     > ,21.94,21.98,21.98,22.01,22.02,22.06,22.11,22.15,22.15,22.16
     > ,22.16,22.16,22.16,22.17,22.17,22.17,22.18,22.18,22.18,22.18
     > ,22.18,22.19,22.19,22.19,22.19,22.19,22.20,22.20,22.18,22.17
     > ,22.16,22.15,22.14,22.14,22.13,22.12,22.12,22.11,22.10,22.09
     > ,22.08,22.06,22.06,22.05,22.06,22.07,22.08,22.09,22.10,22.10
     > ,22.11,22.12,22.13,22.14,22.15,22.15,22.17,22.19,22.21,22.21
     > ,22.23,22.23,22.25,22.27,22.27,22.28,22.29,22.31,22.32,22.33
     > ,22.35,22.37,22.38,22.39,22.40,22.40,22.42,22.43,22.43,22.45
     > ,22.47,22.48,22.50,22.52,22.53,22.55,22.58,22.59,22.60,22.65
     > ,22.68,22.70,22.75,22.80,22.85,22.91,22.97,23.03,23.09,23.11
     > ,23.11,23.15,23.22,23.23,23.29,23.36,23.43,23.50,23.67,23.80
     > ,24.25,24.58,24.61,24.63,24.86,25.07,25.23,25.30,25.13,25.02
     > ,24.77,24.70,24.52,24.13,24.11,23.97,23.58,23.40,23.15,22.95
     > ,22.64,22.50,22.48,22.45,22.40,22.40,22.40,22.40,22.40,22.40
     > ,22.40,22.40,22.44,22.47,22.49,22.52,22.57,22.58,22.67,22.76
     > ,22.76,22.78,22.79,22.80,22.82,22.83,22.86,22.87,22.88,22.89
     > ,22.92,22.95,22.96,22.98,22.99,23.00,23.01,23.03,23.04,23.05
     > ,23.06,23.07,23.09,23.10,23.11,23.13,23.15,23.18,23.21,23.23
     > ,23.24,23.27,23.28,23.30,23.32,23.35,23.37,23.37,23.38,23.39
     > ,23.41,23.42,23.44,23.46,23.49,23.50,23.51,23.52,23.54,23.55
     > ,23.56,23.58,23.58,23.60,23.62,23.62,23.63,23.66,23.67,23.69
     > ,23.71,23.72,23.73,23.75,23.78,23.78,23.79,23.81,23.83,23.85
     > ,23.85,23.86,23.88,23.89,23.92,23.93,23.95,23.96,23.98,24.00
     > ,24.00,24.03,24.05,24.08,24.10,24.13,24.13,24.15,24.18,24.20
     > ,24.30,23.00,23.36,23.80,21.00,24.30,25.20,23.80,25.60,25.60
     > ,25.60,28.20,38.10,21.20,46.10,30.86,20.70,21.40,26.40,33.30
     > ,27.50,21.40,20.83,20.50,14.80,27.60,28.70,61.40,38.71,25.10
     > ,24.08,22.04,20.00,20.85,21.53,22.03,22.58,22.80,19.68,8.76
     > ,32.80,31.95,30.67,29.40,23.50,30.49,44.46,51.44,72.40,59.38
     > ,37.68,29.00,23.30,22.97,22.90,22.63,22.50,22.30,24.35,23.96
     > ,22.94,22.50,22.50,21.71,21.45,20.67,19.88,18.31,17.00,16.40
     > ,15.80,13.70,9.08,7.54,6.00,12.20,17.90,18.10,18.43,20.03
     > ,24.31,28.60,35.72,37.50,46.59,64.77,62.98,43.70,32.90,22.10
     > ,22.25,22.35,22.45,22.60,22.51,22.33,20.80,20.80,20.80,22.40
     > ,20.80,21.27,23.16,24.10,25.00,26.80,27.70,28.60,27.10,25.48
     > ,24.40,22.60,25.00,22.30,21.91,21.13,20.16,19.60,19.24,19.06
     > ,18.19,18.04,17.89,17.80,17.96,18.00,18.00,17.30,17.02,16.74
     > ,15.40,14.44,7.41,4.50,3.60,3.45,3.00,6.75,10.50,10.60
     > ,10.80,11.00,17.55,24.10,22.08,20.06,14.00,13.48,12.83,21.17
     > ,24.00,18.53,15.80,25.70,33.62,34.60,33.35,31.00,24.70,23.65
     > ,19.15,17.77,15.70,16.10,16.70,13.82,11.90,7.10,6.86,6.54
     > ,6.30,10.54,14.78,16.90,14.02,12.58,9.04,11.48,13.10,12.24
     > ,10.95,8.80,9.22,10.20,15.10,11.30,10.55,9.05,14.80,9.20
     > ,11.96,16.10,23.70,18.18,9.90,9.92,10.00,9.50,7.50,8.02
     > ,10.10,9.24,8.38,5.80,8.72,13.10,11.42,8.90,17.60,12.56
     > ,6.30,6.14,6.02,5.90,21.90,11.30,8.85,6.40,7.12,10.00
     > ,5.87,4.10,4.76,5.20,4.99,4.57,13.80,46.12,28.50,24.74
     > ,19.10,16.22,14.30,12.22,9.10,24.20,20.24,14.30,12.70,10.30
     > ,37.60,32.20,19.15,14.50,10.20,12.30,15.80,13.27,17.00,24.50
     > ,18.80,11.70,10.46,8.60,10.60,18.73,21.40,38.30,22.20,9.20
     > ,11.40,106.90,52.90,44.30,27.02,14.00,11.40,10.72,9.70,10.06
     > ,10.30,10.86,11.42,14.58,15.54,28.50,16.00,14.72,12.80,9.40
     > ,8.96,8.52,10.16,11.40,51.30,26.00,14.80,10.30,8.30,8.00
     > ,13.20,12.50,48.20,126.90,63.30,20.30,14.05,7.80,7.20,9.50
     > ,9.14,8.60,8.10,7.70,7.35,8.50,9.50,41.70,70.70,32.30
     > ,13.40,6.70,7.58,29.96,63.20,125.30,48.40,18.80,8.40,6.40
     > ,9.70,22.80,18.04,13.28,7.80,7.30,6.94,6.40,5.80,7.20
     > ,11.30,11.00,9.60,101.90,41.00,37.94,36.92,44.90,38.90,21.62
     > ,15.91,10.56,6.60,6.20,6.30,12.30,5.92,4.50,1.00,0.30
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA N2P_NP/0.00,0.00,0.00,0.00,0.40,0.41,0.36,0.20,0.03,0.04
     > ,0.04,0.06,0.06,0.07,0.07,0.07,0.07,0.07,0.08,0.08
     > ,0.08,0.08,0.08,0.08,0.09,0.09,0.09,0.09,0.10,0.10
     > ,0.10,0.11,0.11,0.11,0.11,0.12,0.12,0.12,0.12,0.12
     > ,0.13,0.13,0.13,0.13,0.14,0.14,0.15,0.15,0.15,0.15
     > ,0.15,0.16,0.16,0.16,0.16,0.16,0.16,0.17,0.17,0.17
     > ,0.17,0.17,0.17,0.18,0.18,0.18,0.18,0.19,0.19,0.20
     > ,0.20,0.21,0.21,0.21,0.22,0.22,0.22,0.22,0.22,0.23
     > ,0.23,0.23,0.24,0.24,0.24,0.24,0.24,0.25,0.25,0.25
     > ,0.25,0.25,0.26,0.26,0.26,0.26,0.27,0.27,0.27,0.27
     > ,0.27,0.28,0.28,0.28,0.28,0.28,0.29,0.29,0.29,0.30
     > ,0.30,0.30,0.31,0.31,0.31,0.32,0.32,0.32,0.32,0.33
     > ,0.33,0.33,0.34,0.34,0.34,0.35,0.35,0.35,0.36,0.36
     > ,0.36,0.37,0.37,0.38,0.38,0.38,0.39,0.39,0.40,0.40
     > ,0.40,0.41,0.41,0.42,0.42,0.42,0.43,0.43,0.44,0.44
     > ,0.44,0.44,0.45,0.45,0.46,0.46,0.46,0.47,0.47,0.47
     > ,0.48,0.48,0.49,0.49,0.49,0.50,0.51,0.51,0.52,0.52
     > ,0.53,0.53,0.53,0.54,0.54,0.55,0.56,0.56,0.57,0.57
     > ,0.58,0.58,0.59,0.60,0.60,0.61,0.61,0.63,0.64,0.64
     > ,0.64,0.66,0.66,0.66,0.67,0.67,0.67,0.68,0.69,0.70
     > ,0.71,0.72,0.73,0.74,0.79,0.83,0.83,0.83,0.83,0.84
     > ,0.84,0.87,0.88,0.89,0.90,0.93,0.94,0.97,0.98,1.00
     > ,1.04,1.07,1.07,1.12,1.13,1.16,1.18,1.24,1.26,1.30
     > ,1.31,1.38,1.40,1.41,1.45,1.46,1.47,1.48,1.49,1.50
     > ,1.52,1.54,1.56,1.56,1.59,1.60,1.61,1.61,1.65,1.66
     > ,1.68,1.71,1.72,1.73,1.74,1.75,1.77,1.80,1.81,1.85
     > ,1.86,1.87,1.88,1.89,1.90,1.91,1.95,1.96,1.99,2.00
     > ,2.01,2.05,2.07,2.08,2.08,2.11,2.12,2.14,2.18,2.19
     > ,2.23,2.23,2.25,2.29,2.33,2.36,2.39,2.41,2.44,2.46
     > ,2.48,2.52,2.52,2.53,2.56,2.59,2.63,2.63,2.64,2.69
     > ,2.72,2.78,2.81,2.82,2.83,2.89,2.90,2.94,2.98,3.01
     > ,3.05,3.06,3.07,3.12,3.14,3.17,3.18,3.18,3.25,3.25
     > ,3.26,3.31,3.34,3.35,3.37,3.39,3.42,3.45,3.46,3.48
     > ,3.49,3.53,3.56,3.59,3.59,3.60,3.60,3.61,3.62,3.63
     > ,3.62,3.62,3.61,3.60,3.59,3.58,3.57,3.55,3.53,3.53
     > ,3.49,3.48,3.47,3.42,3.41,3.39,3.38,3.35,3.31,3.29
     > ,3.27,3.24,3.21,3.20,3.20,3.07,3.06,3.05,3.01,2.99
     > ,2.95,2.93,2.93,2.93,2.92,2.92,2.92,2.92,2.91,2.90
     > ,2.88,2.87,2.80,2.78,2.77,2.77,2.73,2.71,2.70,2.69
     > ,2.67,2.67,2.66,2.62,2.62,2.61,2.59,2.59,2.51,2.49
     > ,2.49,2.45,2.45,2.45,2.45,2.44,2.43,2.43,2.43,2.45
     > ,2.44,2.35,2.34,2.18,2.12,2.12,2.10,2.05,1.98,1.98
     > ,1.91,1.86,1.78,1.76,1.66,1.65,1.56,1.50,1.38,1.23
     > ,1.15,1.07,1.07,1.06,1.06,1.06,1.05,1.05,1.04,1.04
     > ,1.03,1.03,1.02,1.00,0.99,0.97,0.95,0.94,0.92,0.90
     > ,0.90,0.89,0.89,0.87,0.85,0.86,0.86,0.87,0.88,0.88
     > ,0.89,0.90,0.91,0.91,0.92,0.92,0.92,0.93,0.93,0.93
     > ,0.93,0.94,0.94,0.94,0.94,0.94,0.95,0.95,0.95,0.94
     > ,0.94,0.94,0.94,0.93,0.93,0.93,0.93,0.92,0.92,0.92
     > ,0.92,0.92,0.91,0.91,0.91,0.90,0.90,0.90,0.91,0.92
     > ,0.93,0.93,0.94,0.94,0.95,0.95,0.95,0.96,0.97,0.97
     > ,0.98,0.99,0.99,1.00,1.01,1.02,1.04,1.05,1.06,1.06
     > ,1.07,1.08,1.10,1.11,1.12,1.12,1.13,1.15,1.16,1.16
     > ,1.17,1.17,1.18,1.20,1.20,1.20,1.21,1.22,1.23,1.24
     > ,1.25,1.22,1.21,1.19,1.18,1.18,1.16,1.14,1.13,1.11
     > ,1.09,1.07,1.04,1.02,1.00,0.97,0.93,0.92,0.90,0.85
     > ,0.81,0.80,0.75,0.70,0.65,0.59,0.53,0.47,0.41,0.39
     > ,0.39,0.35,0.30,0.29,0.24,0.19,0.13,0.08,0.05,0.03
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA N2P_ION/0.80,0.96,0.97,0.99,1.01,1.02,0.91,0.51,0.08,0.10
     > ,0.11,0.15,0.15,0.17,0.17,0.17,0.18,0.18,0.19,0.19
     > ,0.20,0.20,0.21,0.21,0.23,0.23,0.24,0.24,0.24,0.25
     > ,0.26,0.27,0.28,0.28,0.29,0.29,0.30,0.30,0.31,0.31
     > ,0.32,0.33,0.33,0.34,0.35,0.36,0.37,0.38,0.38,0.39
     > ,0.39,0.40,0.40,0.41,0.41,0.41,0.42,0.42,0.43,0.43
     > ,0.44,0.44,0.45,0.45,0.46,0.46,0.47,0.48,0.50,0.51
     > ,0.52,0.53,0.53,0.54,0.56,0.56,0.57,0.57,0.57,0.59
     > ,0.60,0.60,0.61,0.62,0.62,0.62,0.63,0.63,0.64,0.65
     > ,0.65,0.66,0.66,0.66,0.68,0.68,0.69,0.69,0.70,0.70
     > ,0.71,0.71,0.72,0.73,0.73,0.74,0.74,0.75,0.76,0.78
     > ,0.78,0.79,0.80,0.81,0.82,0.82,0.83,0.84,0.85,0.85
     > ,0.86,0.87,0.88,0.89,0.89,0.90,0.91,0.93,0.94,0.94
     > ,0.95,0.96,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.04
     > ,1.05,1.06,1.07,1.10,1.11,1.12,1.12,1.13,1.15,1.16
     > ,1.16,1.17,1.19,1.20,1.20,1.22,1.23,1.24,1.24,1.25
     > ,1.26,1.28,1.28,1.29,1.31,1.32,1.34,1.36,1.39,1.39
     > ,1.39,1.40,1.41,1.42,1.44,1.45,1.49,1.50,1.51,1.53
     > ,1.54,1.55,1.59,1.60,1.62,1.62,1.64,1.68,1.72,1.73
     > ,1.73,1.77,1.78,1.82,1.84,1.87,1.95,1.97,2.00,2.02
     > ,2.06,2.09,2.11,2.15,2.27,2.36,2.37,2.38,2.39,2.41
     > ,2.42,2.56,2.61,2.62,2.68,2.77,2.82,2.95,2.98,3.02
     > ,3.14,3.21,3.22,3.32,3.35,3.42,3.46,3.60,3.63,3.71
     > ,3.73,3.92,3.97,3.99,4.11,4.14,4.16,4.18,4.22,4.24
     > ,4.30,4.35,4.40,4.41,4.48,4.51,4.52,4.54,4.64,4.64
     > ,4.69,4.77,4.79,4.81,4.84,4.86,4.89,4.96,4.98,5.05
     > ,5.08,5.10,5.13,5.14,5.16,5.17,5.27,5.29,5.36,5.38
     > ,5.41,5.50,5.55,5.57,5.59,5.66,5.68,5.73,5.83,5.84
     > ,5.94,5.95,6.01,6.09,6.20,6.28,6.36,6.40,6.49,6.52
     > ,6.57,6.68,6.69,6.71,6.78,6.85,6.96,6.97,6.98,7.10
     > ,7.17,7.30,7.38,7.40,7.41,7.57,7.58,7.69,7.77,7.85
     > ,7.96,7.97,8.01,8.13,8.17,8.26,8.28,8.29,8.45,8.47
     > ,8.49,8.60,8.68,8.71,8.76,8.84,8.92,9.02,9.07,9.12
     > ,9.16,9.30,9.38,9.49,9.50,9.54,9.60,9.68,9.73,9.80
     > ,9.84,9.85,9.88,9.90,9.93,9.98,10.02,10.06,10.08,10.09
     > ,10.14,10.16,10.18,10.21,10.21,10.22,10.23,10.25,10.27,10.28
     > ,10.31,10.35,10.38,10.40,10.40,10.49,10.50,10.51,10.54,10.56
     > ,10.59,10.61,10.62,10.63,10.65,10.67,10.68,10.69,10.73,10.78
     > ,10.81,10.82,10.90,10.92,10.95,10.95,11.04,11.07,11.10,11.12
     > ,11.16,11.17,11.19,11.27,11.30,11.32,11.48,11.50,11.54,11.67
     > ,11.70,12.02,12.37,12.45,12.46,12.66,12.72,12.73,12.77,13.13
     > ,13.55,13.95,13.98,14.38,14.82,14.83,14.89,15.04,15.27,15.28
     > ,15.63,15.82,16.18,16.25,16.58,16.61,16.91,17.08,17.53,18.03
     > ,19.05,20.05,20.07,20.18,20.23,20.25,20.38,20.55,20.64,20.74
     > ,20.83,20.93,21.02,21.10,21.19,21.27,21.35,21.44,21.52,21.60
     > ,21.62,21.66,21.68,21.77,21.85,21.93,22.01,22.09,22.17,22.25
     > ,22.32,22.40,22.48,22.56,22.64,22.66,22.69,22.73,22.78,22.82
     > ,22.87,22.92,22.92,22.95,22.96,23.01,23.05,23.10,23.10,23.10
     > ,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10
     > ,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.10,23.09
     > ,23.09,23.09,23.08,23.08,23.08,23.08,23.07,23.07,23.07,23.07
     > ,23.06,23.06,23.05,23.05,23.07,23.09,23.12,23.14,23.16,23.16
     > ,23.18,23.20,23.23,23.25,23.27,23.27,23.30,23.34,23.37,23.38
     > ,23.40,23.40,23.44,23.46,23.47,23.48,23.50,23.53,23.55,23.57
     > ,23.60,23.59,23.59,23.58,23.58,23.58,23.57,23.57,23.57,23.56
     > ,23.55,23.55,23.54,23.53,23.53,23.52,23.51,23.50,23.50,23.50
     > ,23.50,23.50,23.50,23.50,23.50,23.50,23.50,23.50,23.50,23.50
     > ,23.50,23.50,23.52,23.52,23.53,23.55,23.56,23.58,23.73,23.83
     > ,24.25,24.58,24.61,24.63,24.86,25.07,25.23,25.30,25.13,25.02
     > ,24.77,24.70,24.52,24.13,24.11,23.97,23.58,23.40,23.15,22.95
     > ,22.64,22.50,22.48,22.45,22.40,22.40,22.40,22.40,22.40,22.40
     > ,22.40,22.40,22.44,22.47,22.49,22.52,22.57,22.58,22.67,22.76
     > ,22.76,22.78,22.79,22.80,22.82,22.83,22.86,22.87,22.88,22.89
     > ,22.92,22.95,22.96,22.98,22.99,23.00,23.01,23.03,23.04,23.05
     > ,23.06,23.07,23.09,23.10,23.11,23.13,23.15,23.18,23.21,23.23
     > ,23.24,23.27,23.28,23.30,23.32,23.35,23.37,23.37,23.38,23.39
     > ,23.41,23.42,23.44,23.46,23.49,23.50,23.51,23.52,23.54,23.55
     > ,23.56,23.58,23.58,23.60,23.62,23.62,23.63,23.66,23.67,23.69
     > ,23.71,23.72,23.73,23.75,23.78,23.78,23.79,23.81,23.83,23.85
     > ,23.85,23.86,23.88,23.89,23.92,23.93,23.95,23.96,23.98,24.00
     > ,24.00,24.03,24.05,24.08,24.10,24.13,24.13,24.15,24.18,24.20
     > ,24.30,23.00,23.36,23.80,21.00,24.30,25.20,23.80,25.60,25.60
     > ,25.60,28.20,38.10,21.20,46.10,30.86,20.70,21.40,26.40,33.30
     > ,27.50,21.40,20.83,20.50,14.80,27.60,28.70,61.40,38.71,25.10
     > ,24.08,22.04,20.00,20.85,21.53,22.03,22.58,22.80,19.68,8.76
     > ,32.80,31.95,30.67,29.40,23.50,30.49,44.46,51.44,72.40,59.38
     > ,37.68,29.00,23.30,22.97,22.90,22.63,22.50,22.30,24.35,23.96
     > ,22.94,22.50,22.50,21.71,21.45,20.67,19.88,18.31,17.00,16.40
     > ,15.80,13.70,9.08,7.54,6.00,12.20,17.90,18.10,18.43,20.03
     > ,24.31,28.60,35.72,37.50,46.59,64.77,62.98,43.70,32.90,22.10
     > ,22.25,22.35,22.45,22.60,22.51,22.33,20.80,20.80,20.80,22.40
     > ,20.80,21.27,23.16,24.10,25.00,26.80,27.70,28.60,27.10,25.48
     > ,24.40,22.60,25.00,22.30,21.91,21.13,20.16,19.60,19.24,19.06
     > ,18.19,18.04,17.89,17.80,17.96,18.00,18.00,17.30,17.02,16.74
     > ,15.40,14.44,7.41,4.50,3.60,3.45,3.00,6.75,10.50,10.60
     > ,10.80,11.00,17.55,24.10,22.08,20.06,14.00,13.48,12.83,21.17
     > ,24.00,18.53,15.80,25.70,33.62,34.60,33.35,31.00,24.70,23.65
     > ,19.15,17.77,15.70,16.10,16.70,13.82,11.90,7.10,6.86,6.54
     > ,6.30,10.54,14.78,16.90,14.02,12.58,9.04,11.48,13.10,12.24
     > ,10.95,8.80,9.22,10.20,15.10,11.30,10.55,9.05,14.80,9.20
     > ,11.96,16.10,23.70,18.18,9.90,9.92,10.00,9.50,7.50,8.02
     > ,10.10,9.24,8.38,5.80,8.72,13.10,11.42,8.90,17.60,12.56
     > ,6.30,6.14,6.02,5.90,21.90,11.30,8.85,6.40,7.12,10.00
     > ,5.87,4.10,4.76,5.20,4.99,4.57,13.80,46.12,28.50,24.74
     > ,19.10,16.22,14.30,12.22,9.10,24.20,20.24,14.30,12.70,10.30
     > ,37.60,32.20,19.15,14.50,10.20,12.30,15.80,13.27,17.00,24.50
     > ,18.80,11.70,10.46,8.60,10.60,18.73,21.40,38.30,22.20,9.20
     > ,11.40,106.90,52.90,44.30,27.02,14.00,11.40,10.72,9.70,10.06
     > ,10.30,10.86,11.42,14.58,15.54,28.50,16.00,14.72,12.80,9.40
     > ,8.96,8.52,10.16,11.40,51.30,26.00,14.80,10.30,8.30,8.00
     > ,13.20,12.50,48.20,126.90,63.30,20.30,14.05,7.80,7.20,9.50
     > ,9.14,8.60,8.10,7.70,7.35,8.50,9.50,41.70,70.70,32.30
     > ,13.40,6.70,7.58,29.96,63.20,125.30,48.40,18.80,8.40,6.40
     > ,9.70,22.80,18.04,13.28,7.80,7.30,6.94,6.40,5.80,7.20
     > ,11.30,11.00,9.60,101.90,41.00,37.94,36.92,44.90,38.90,21.62
     > ,15.91,10.56,6.60,6.20,6.30,12.30,5.92,4.50,1.00,0.30
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA O2_ABS/0.07,0.13,0.13,0.14,0.14,0.15,0.15,0.16,0.16,0.19
     > ,0.21,0.28,0.28,0.31,0.31,0.31,0.32,0.33,0.34,0.35
     > ,0.36,0.36,0.38,0.39,0.42,0.43,0.44,0.44,0.45,0.47
     > ,0.48,0.50,0.52,0.53,0.54,0.54,0.55,0.56,0.58,0.59
     > ,0.60,0.61,0.62,0.62,0.65,0.67,0.68,0.70,0.70,0.72
     > ,0.73,0.74,0.75,0.75,0.76,0.76,0.77,0.78,0.79,0.80
     > ,0.82,0.84,0.84,0.86,0.89,0.89,0.90,0.93,0.94,0.96
     > ,0.98,0.99,1.00,1.00,1.03,1.04,1.04,1.06,1.06,1.09
     > ,1.11,1.12,1.14,1.14,1.15,1.16,1.17,1.18,1.19,1.21
     > ,1.21,1.22,1.23,1.24,1.27,1.28,1.29,1.30,1.31,1.32
     > ,1.33,1.34,1.36,1.36,1.38,1.39,1.40,1.41,1.43,1.45
     > ,1.46,1.47,1.49,1.50,1.51,1.53,1.53,1.55,1.56,1.57
     > ,1.58,1.60,1.61,1.62,1.64,1.65,1.67,1.69,1.71,1.71
     > ,1.73,1.74,1.77,1.79,1.80,1.82,1.83,1.85,1.86,1.87
     > ,1.88,1.91,1.92,1.96,1.98,1.99,2.00,2.02,2.04,2.05
     > ,2.06,2.08,2.10,2.11,2.13,2.15,2.16,2.18,2.18,2.20
     > ,2.21,2.24,2.25,2.27,2.29,2.31,2.34,2.37,2.41,2.42
     > ,2.43,2.44,2.46,2.47,2.50,2.52,2.57,2.59,2.60,2.65
     > ,2.66,2.68,2.73,2.75,2.78,2.79,2.81,2.89,2.94,2.95
     > ,2.96,3.02,3.04,3.08,3.10,3.15,3.24,3.27,3.32,3.37
     > ,3.43,3.49,3.52,3.60,3.78,3.94,3.95,3.97,3.99,4.02
     > ,4.03,4.30,4.38,4.40,4.50,4.65,4.73,4.93,4.99,5.05
     > ,5.21,5.32,5.33,5.47,5.52,5.62,5.68,5.87,5.91,6.03
     > ,6.05,6.31,6.37,6.39,6.53,6.58,6.60,6.63,6.68,6.70
     > ,6.77,6.83,6.88,6.89,6.99,7.01,7.03,7.04,7.15,7.15
     > ,7.21,7.28,7.31,7.33,7.36,7.39,7.42,7.51,7.53,7.61
     > ,7.65,7.67,7.70,7.72,7.74,7.76,7.87,7.89,7.98,8.00
     > ,8.04,8.15,8.20,8.22,8.24,8.32,8.35,8.40,8.51,8.51
     > ,8.61,8.62,8.68,8.76,8.86,8.94,9.01,9.05,9.13,9.16
     > ,9.21,9.30,9.31,9.33,9.39,9.45,9.54,9.55,9.57,9.67
     > ,9.73,9.85,9.92,9.94,9.95,10.07,10.08,10.17,10.23,10.30
     > ,10.40,10.41,10.45,10.56,10.60,10.68,10.70,10.71,10.86,10.88
     > ,10.90,11.00,11.07,11.10,11.15,11.22,11.31,11.41,11.45,11.50
     > ,11.54,11.69,11.77,11.89,11.90,11.96,12.06,12.17,12.25,12.34
     > ,12.43,12.46,12.52,12.55,12.62,12.73,12.80,12.90,12.98,13.00
     > ,13.14,13.20,13.25,13.37,13.40,13.44,13.47,13.55,13.65,13.70
     > ,13.80,14.00,14.12,14.18,14.20,14.65,14.70,14.74,14.86,14.91
     > ,15.04,15.10,15.13,15.15,15.19,15.24,15.26,15.28,15.37,15.50
     > ,15.56,15.60,15.79,15.85,15.90,15.91,16.09,16.14,16.20,16.23
     > ,16.28,16.30,16.34,16.45,16.48,16.51,16.68,16.70,16.72,16.80
     > ,16.81,17.00,17.10,17.12,17.12,17.18,17.20,17.20,17.21,17.25
     > ,17.30,17.40,17.41,17.50,17.65,17.65,17.67,17.72,17.80,17.80
     > ,17.88,17.92,18.00,18.03,18.18,18.19,18.32,18.40,18.59,18.80
     > ,19.20,19.59,19.60,19.65,19.67,19.68,19.73,19.80,19.84,19.88
     > ,19.92,19.96,20.00,20.03,20.06,20.09,20.12,20.15,20.18,20.21
     > ,20.22,20.23,20.24,20.27,20.30,20.34,20.38,20.42,20.46,20.50
     > ,20.54,20.58,20.62,20.66,20.70,20.71,20.73,20.76,20.79,20.82
     > ,20.85,20.88,20.88,20.90,20.91,20.94,20.97,21.00,21.02,21.04
     > ,21.08,21.09,21.11,21.12,21.14,21.16,21.20,21.20,21.24,21.25
     > ,21.27,21.28,21.30,21.32,21.34,21.36,21.40,21.41,21.43,21.46
     > ,21.48,21.49,21.52,21.53,21.54,21.55,21.56,21.58,21.60,21.61
     > ,21.64,21.67,21.68,21.70,21.74,21.78,21.82,21.86,21.90,21.91
     > ,21.94,21.98,22.02,22.06,22.09,22.10,22.15,22.20,22.25,22.26
     > ,22.29,22.30,22.35,22.40,22.40,22.42,22.45,22.50,22.52,22.55
     > ,22.60,22.63,22.64,22.67,22.68,22.68,22.70,22.72,22.73,22.76
     > ,22.79,22.80,22.84,22.86,22.88,22.92,22.96,22.98,23.00,23.06
     > ,23.10,23.12,23.18,23.24,23.30,23.36,23.42,23.48,23.54,23.56
     > ,23.56,23.60,23.64,23.64,23.68,23.72,23.76,23.80,23.92,24.00
     > ,24.28,24.50,24.53,24.54,24.73,24.90,25.25,25.40,25.28,25.20
     > ,25.90,25.90,25.90,26.04,26.05,26.10,26.10,26.10,26.04,26.00
     > ,25.86,25.80,25.66,25.50,24.80,24.61,22.88,22.72,22.40,21.80
     > ,19.30,19.40,20.80,21.80,23.80,27.70,28.49,28.60,30.10,27.12
     > ,26.80,26.40,27.96,28.54,29.91,29.73,25.68,24.21,23.10,27.00
     > ,28.43,28.57,28.60,25.40,24.20,26.43,28.66,34.60,30.82,28.30
     > ,29.80,29.06,25.71,24.96,23.10,26.53,32.70,29.57,26.72,25.30
     > ,26.06,27.77,29.00,25.72,27.60,30.60,32.40,32.14,31.34,30.55
     > ,28.69,27.11,26.05,23.40,29.11,31.40,30.09,28.79,25.30,26.34
     > ,27.90,26.08,25.30,27.58,26.17,25.30,25.59,26.57,26.96,27.47
     > ,28.33,28.51,29.00,26.59,21.76,20.80,21.93,25.30,24.90,29.45
     > ,30.10,29.57,28.68,27.79,26.01,25.30,26.42,26.99,28.30,29.80
     > ,29.71,29.29,28.86,28.43,28.00,27.57,27.44,27.14,26.71,26.29
     > ,26.16,25.86,25.69,25.47,25.00,24.57,24.31,24.19,24.01,23.80
     > ,23.74,23.45,23.37,23.16,22.87,22.70,20.80,21.44,21.23,20.86
     > ,20.56,19.82,19.30,19.63,19.96,20.61,21.60,21.17,20.10,21.00
     > ,21.60,20.10,21.20,20.52,19.97,19.43,18.75,18.47,18.20,23.80
     > ,22.11,20.99,19.30,21.55,22.03,21.90,22.30,21.77,20.19,18.60
     > ,21.90,21.02,19.70,17.50,18.38,21.91,23.67,26.32,27.20,21.95
     > ,20.11,17.57,19.54,23.80,23.25,21.62,19.98,16.70,19.26,20.80
     > ,17.50,25.04,22.63,21.27,19.92,18.57,19.42,28.53,31.56,34.60
     > ,31.53,28.45,20.20,18.20,21.27,27.43,30.50,19.30,23.61,27.92
     > ,30.50,31.05,31.60,33.10,34.60,29.43,25.12,23.40,24.37,27.27
     > ,30.17,31.13,35.00,31.00,27.00,19.00,20.33,21.67,28.33,27.44
     > ,26.00,34.75,45.69,56.64,63.20,45.52,24.90,25.36,26.05,26.40
     > ,25.70,26.53,25.70,29.39,39.24,41.70,32.40,35.70,36.80,37.90
     > ,35.28,33.54,35.00,33.62,30.85,30.38,29.00,30.50,28.36,27.50
     > ,29.80,28.71,27.35,25.99,25.44,24.90,30.48,34.20,30.09,26.80
     > ,26.94,27.50,27.05,25.92,26.00,25.30,26.68,27.50,25.85,25.57
     > ,33.80,37.20,42.30,49.10,36.43,30.10,30.18,30.38,30.50,29.01
     > ,27.90,29.80,29.00,29.97,32.88,34.33,33.19,31.60,37.23,42.86
     > ,51.30,39.67,32.70,33.61,34.26,34.91,35.30,34.38,33.47,32.71
     > ,32.40,34.20,31.62,29.55,26.45,25.42,25.70,24.77,21.03,20.10
     > ,21.30,21.60,20.86,18.64,17.90,18.33,18.62,19.06,19.78,20.36
     > ,19.14,18.60,19.90,21.20,18.40,15.60,17.65,19.70,20.52,23.80
     > ,19.39,17.50,19.00,17.74,16.79,14.90,19.35,20.97,23.40,22.52
     > ,21.20,19.88,19.00,19.30,19.00,18.50,18.30,18.00,17.80,17.50
     > ,18.85,19.30,18.68,17.98,17.10,18.99,19.62,20.25,21.20,20.49
     > ,19.30,19.63,19.60,19.30,21.16,21.90,20.96,20.33,19.39,18.13
     > ,18.10,19.90,21.70,22.30,21.54,20.78,19.51,19.00,19.55,20.10
     > ,19.55,19.01,18.46,17.37,17.10,19.88,19.35,17.90,20.38,17.20
     > ,15.20,18.73,19.70,20.83,23.64,24.00,23.80,26.37,24.20,20.95
     > ,17.70,14.45,15.87,16.90,18.45,21.03,23.62,26.20,28.78,26.56
     > ,25.19,25.07,26.48,27.33,27.62,25.22,23.43,18.97,17.18,14.50
     > ,16.75,19.00,20.80,20.50,21.80,23.10,23.97,25.70,23.80,24.30
     > ,25.55,26.80,23.25,19.70,22.70,25.20,25.32,24.20,26.95,26.80
     > ,27.55,28.30,25.57,23.93,22.84,21.20,21.81,24.25,27.90,24.20
     > ,25.09,26.71,29.37,33.06,29.04,23.10,23.40,21.90,27.61,31.60
     > ,28.00,35.16,39.80,36.80,33.80,30.80,27.80,30.00,34.20,30.56
     > ,26.00,33.22,37.07,38.71,40.44,44.41,47.66,46.92,45.31,41.90
     > ,39.06,36.48,34.37,27.90,34.98,36.40,32.55,28.71,26.15,22.30
     > ,24.92,31.46,38.00,40.15,42.57,44.54,47.94,50.92,52.51,55.00
     > ,47.69,37.81,29.40,30.80,31.73,34.07,35.00,30.90,24.07,18.60
     > ,25.59,31.42,36.31,40.34,43.07,45.28,37.81,30.80,25.16,20.10
     > ,21.11,22.26,25.85,29.44,31.60,29.52,24.32,19.12,16.00,17.29
     > ,20.53,27.01,28.30,17.37,20.10,24.20,20.10,22.64,29.00,18.31
     > ,11.90,11.98,12.18,12.30,11.86,11.20,18.52,22.70,22.30,23.10
     > ,21.05,10.80,25.40,32.70,28.52,27.12,18.76,15.97,11.79,10.50
     > ,10.80,10.36,10.18,10.00,14.73,17.10,14.35,12.38,11.20,14.53
     > ,24.50,23.40,19.89,17.11,15.99,14.32,11.54,8.76,9.16,10.37
     > ,11.58,12.30,11.72,10.29,12.65,15.95,18.60,11.35,7.40,7.58
     > ,7.80,7.51,7.40,7.89,8.71,9.70,9.33,8.87,8.60,9.17
     > ,9.71,10.60,12.03,12.60,11.68,10.16,8.63,7.10,7.73,8.36
     > ,8.61,8.99,9.62,10.00,9.31,8.78,7.02,7.05,6.70,7.01
     > ,7.17,7.40,7.66,7.93,8.46,8.56,8.98,9.30,8.81,8.56
     > ,8.31,7.57,6.96,6.34,5.72,6.03,6.58,7.12,7.66,8.20
     > ,9.23,9.49,10.00,9.31,8.73,8.15,7.57,6.99,6.41,5.83
     > ,5.60,6.41,7.75,8.29,9.10,10.44,11.79,12.38,12.74,12.04
     > ,11.58,11.08,10.59,10.26,9.43,8.60,7.78,6.95,6.12,5.79
     > ,5.30,4.80,5.85,8.47,11.09,13.71,16.33,17.90,17.11,15.12
     > ,13.53,11.15,9.17,7.18,5.20,6.00,7.50,9.00,10.50,9.76
     > ,8.46,7.16,5.86,8.09,11.20,9.64,9.02,8.09,6.53,5.60
     > ,5.84,6.44,7.04,7.28,7.03,6.10,5.17,4.80,5.62,6.99
     > ,8.35,8.90,10.40,12.90,12.58,11.76,9.70,10.31,10.62,10.80
     > ,10.28,9.00,7.71,6.42,5.13,4.10,6.54,8.16,10.60,12.63
     > ,14.66,16.69,16.15,15.20,17.04,15.23,12.96,11.78,10.70,9.24
     > ,8.43,6.16,4.80,5.17,6.10,7.03,7.40,6.30,4.47,8.93
     > ,14.96,20.99,23.40,21.49,18.30,15.11,12.56,8.74,6.19,3.00
     > ,4.39,5.73,6.97,8.21,8.96,9.45,8.90,11.80,16.15,20.50
     > ,23.40,18.22,13.03,7.85,5.77,3.70,3.90,4.10,3.46,3.62
     > ,4.02,3.70,22.28,23.72,19.91,12.30,19.54,28.60,23.76,20.70
     > ,19.73,15.70,11.67,7.64,5.22,3.61,7.36,13.07,18.77,24.47
     > ,27.89,29.03,35.88,41.58,45.00,43.40,39.40,35.40,31.40,27.40
     > ,23.40,20.20,15.40,13.00,11.40,7.40,3.40,9.52,26.81,44.10
     > ,56.20,49.97,39.58,29.20,18.82,13.83,8.43,2.20,4.08,8.76
     > ,18.14,27.51,36.89,46.26,54.70,35.43,29.00,35.30,29.83,24.37
     > ,22.18,18.90,15.62,13.43,7.97,2.50,5.95,9.39,12.84,15.60
     > ,9.67,6.70,9.98,26.39,42.80,51.00,46.04,41.08,36.12,31.16
     > ,26.20,16.29,11.33,6.37,2.40,10.74,17.70,24.65,31.60,42.80
     > ,28.85,17.23,5.60,18.85,26.80,25.86,23.52,21.17,18.82,16.48
     > ,14.13,11.78,9.44,7.09,4.75,2.40,10.20,18.01,25.81,33.61
     > ,41.42,46.10,41.75,30.88,20.02,9.15,4.80,5.91,7.40,6.56
     > ,5.10,3.00,4.80,3.30,1.50,2.76,4.15,10.79,14.11,17.43
     > ,23.40,21.60,24.50,21.39,15.16,8.93,2.70,1.45,1.46,1.48
     > ,1.49,2.69,3.90,5.10,6.30,5.60,6.30,5.41,3.17,1.38
     > ,1.46,1.82,1.44,1.52,1.45,1.43,1.34,1.30,1.15,1.08
     > ,1.23,1.34,1.12,1.48,1.75,1.19,1.38,1.08,1.60,0.97
     > ,1.19,1.52,1.08,1.19,1.41,1.28,1.19,1.60,1.40,1.64
     > ,1.47,1.30,1.64,1.86,1.43,1.00,1.45,1.79,1.64,1.52
     > ,1.11/

       DATA O2P/0.07,0.13,0.13,0.14,0.14,0.15,0.15,0.16,0.16,0.19
     > ,0.21,0.28,0.28,0.31,0.31,0.31,0.32,0.33,0.34,0.35
     > ,0.36,0.36,0.38,0.39,0.42,0.43,0.43,0.44,0.45,0.47
     > ,0.48,0.50,0.52,0.53,0.54,0.54,0.55,0.56,0.58,0.59
     > ,0.60,0.61,0.62,0.62,0.65,0.67,0.68,0.70,0.70,0.72
     > ,0.73,0.74,0.75,0.75,0.76,0.76,0.77,0.78,0.79,0.80
     > ,0.82,0.84,0.84,0.86,0.89,0.89,0.90,0.93,0.94,0.96
     > ,0.98,0.99,1.00,1.00,1.03,1.04,1.04,1.06,1.06,1.09
     > ,1.11,1.12,1.14,1.14,1.15,1.16,1.17,1.18,1.19,1.21
     > ,1.21,1.22,1.23,1.24,1.27,1.28,1.29,1.30,1.31,1.32
     > ,1.33,1.34,1.36,1.36,1.38,1.39,1.40,1.41,1.43,1.45
     > ,1.46,1.47,1.49,1.50,1.51,1.53,1.53,1.55,1.56,1.57
     > ,1.58,1.60,1.61,1.62,1.64,1.65,1.67,1.69,1.71,1.71
     > ,1.73,1.74,1.77,1.79,1.80,1.82,1.83,1.85,1.86,1.87
     > ,1.88,1.91,1.92,1.96,1.98,1.99,2.00,2.02,2.04,2.05
     > ,2.06,2.08,2.10,2.11,2.13,2.15,2.16,2.18,2.18,2.20
     > ,2.21,2.24,2.25,2.27,2.29,2.31,2.34,2.37,2.41,2.42
     > ,2.43,2.44,2.46,2.47,2.50,2.52,2.57,2.59,2.60,2.65
     > ,2.66,2.68,2.73,2.75,2.78,2.79,2.81,2.89,2.94,2.95
     > ,2.96,3.02,3.04,3.08,3.10,3.15,1.13,1.15,1.18,1.21
     > ,1.24,1.28,1.30,1.35,1.47,1.57,1.58,1.60,1.61,1.63
     > ,1.64,1.84,1.91,1.92,2.00,2.12,2.17,2.30,2.34,2.38
     > ,2.49,2.56,2.56,2.66,2.69,2.75,2.78,2.91,2.94,3.01
     > ,3.03,3.21,3.26,3.27,3.38,3.41,3.43,3.45,3.49,3.50
     > ,3.56,3.62,3.67,3.67,3.76,3.78,3.80,3.81,3.90,3.91
     > ,3.95,4.03,4.05,4.07,4.09,4.11,4.13,4.18,4.20,4.25
     > ,4.28,4.29,4.31,4.32,4.33,4.34,4.42,4.43,4.49,4.51
     > ,4.53,4.60,4.63,4.65,4.66,4.72,4.74,4.77,4.84,4.85
     > ,4.92,4.92,4.96,5.02,5.09,5.15,5.20,5.24,5.30,5.33
     > ,5.36,5.44,5.44,5.46,5.51,5.55,5.63,5.64,5.65,5.71
     > ,5.75,5.83,5.88,5.89,5.90,5.99,5.99,6.05,6.10,6.14
     > ,6.22,6.23,6.25,6.35,6.38,6.44,6.46,6.46,6.59,6.60
     > ,6.62,6.70,6.75,6.78,6.82,6.87,6.93,7.00,7.03,7.06
     > ,7.09,7.20,7.26,7.34,7.35,7.40,7.48,7.57,7.64,7.72
     > ,7.79,7.81,7.87,7.89,7.95,8.04,8.10,8.16,8.21,8.22
     > ,8.32,8.35,8.38,8.46,8.48,8.51,8.52,8.57,8.64,8.67
     > ,8.72,8.81,8.86,8.89,8.90,9.09,9.11,9.14,9.22,9.25
     > ,9.34,9.38,9.40,9.42,9.44,9.48,9.49,9.50,9.56,9.65
     > ,9.70,9.73,9.89,9.94,9.98,9.99,10.14,10.19,10.24,10.27
     > ,10.31,10.33,10.36,10.46,10.48,10.51,10.66,10.68,10.72,10.86
     > ,10.88,11.22,11.45,11.50,11.51,11.64,11.68,11.69,11.71,11.89
     > ,12.10,12.33,12.35,12.56,12.73,12.73,12.75,12.81,12.89,12.89
     > ,13.05,13.14,13.30,13.34,13.55,13.57,13.76,13.87,13.97,14.07
     > ,14.40,14.80,14.81,14.85,14.87,14.88,14.93,15.00,15.04,15.08
     > ,15.12,15.15,15.19,15.23,15.26,15.29,15.32,15.36,15.39,15.42
     > ,15.43,15.45,15.46,15.49,15.52,15.55,15.58,15.61,15.64,15.67
     > ,15.70,15.73,15.76,15.79,15.82,15.84,15.86,15.90,15.94,15.98
     > ,16.01,16.05,16.06,16.08,16.09,16.13,16.17,16.21,16.23,16.25
     > ,16.28,16.29,16.31,16.32,16.34,16.36,16.40,16.40,16.43,16.44
     > ,16.46,16.47,16.49,16.51,16.53,16.55,16.58,16.59,16.60,16.62
     > ,16.63,16.64,16.66,16.67,16.68,16.68,16.69,16.70,16.71,16.72
     > ,16.74,16.76,16.76,16.78,16.81,16.85,16.89,16.92,16.96,16.97
     > ,16.99,17.03,17.07,17.10,17.13,17.14,17.16,17.18,17.20,17.20
     > ,17.21,17.22,17.24,17.25,17.25,17.26,17.27,17.29,17.30,17.31
     > ,17.33,17.36,17.37,17.39,17.41,17.41,17.43,17.44,17.45,17.48
     > ,17.51,17.52,17.56,17.58,17.59,17.63,17.67,17.69,17.71,17.75
     > ,17.77,17.78,17.82,17.86,17.90,17.94,17.98,18.02,18.05,18.06
     > ,18.07,18.09,18.13,18.13,18.17,18.21,18.25,18.28,18.40,18.48
     > ,18.81,19.08,19.14,19.17,19.60,19.98,20.50,20.72,20.92,21.06
     > ,21.34,21.48,21.67,21.60,21.60,21.58,21.58,21.59,21.55,21.52
     > ,21.41,21.36,21.25,21.13,20.56,20.40,18.98,18.84,18.58,18.08
     > ,16.12,16.12,17.09,17.76,19.98,24.56,25.54,25.67,26.62,23.63
     > ,23.32,23.01,24.43,24.97,26.79,26.65,23.81,22.72,21.66,25.29
     > ,26.55,26.59,26.60,23.56,22.43,24.47,26.50,31.98,28.47,26.13
     > ,27.49,26.78,23.59,22.89,21.13,24.20,29.70,26.71,24.02,22.68
     > ,23.85,25.71,26.91,23.90,25.73,28.63,30.37,30.14,29.43,28.73
     > ,27.04,25.58,24.61,22.17,27.65,29.86,28.64,27.43,24.20,25.22
     > ,26.75,25.09,24.38,26.43,24.98,24.12,24.34,25.08,25.37,25.75
     > ,26.37,26.50,26.85,24.65,20.22,19.34,20.39,23.61,23.20,27.41
     > ,28.01,27.51,26.66,25.81,24.12,23.43,24.43,24.93,26.08,27.42
     > ,27.37,27.09,26.81,26.53,26.25,25.96,25.87,25.66,25.37,25.07
     > ,24.98,24.77,24.65,24.49,23.97,23.50,23.22,23.11,22.96,22.77
     > ,22.72,22.47,22.40,22.22,21.97,21.82,20.01,20.63,20.44,20.10
     > ,20.23,19.32,18.69,18.97,19.25,19.81,20.64,20.19,19.08,19.99
     > ,20.59,19.24,20.36,19.16,18.22,17.95,17.59,17.45,17.30,23.11
     > ,21.67,20.69,19.20,20.32,20.84,20.93,21.77,21.13,19.25,17.43
     > ,20.97,20.21,19.06,17.11,18.01,21.64,23.48,26.27,27.20,21.95
     > ,20.11,17.34,18.99,22.85,22.22,20.35,18.53,15.03,16.98,18.10
     > ,15.75,23.46,21.08,19.75,18.43,17.12,17.83,25.88,28.52,31.14
     > ,28.49,25.82,18.58,16.74,19.49,24.92,27.60,17.27,20.98,24.64
     > ,26.80,27.21,27.61,28.80,29.97,25.27,21.42,19.89,20.71,23.18
     > ,25.64,26.46,29.75,26.35,22.95,16.15,17.28,18.42,24.08,23.75
     > ,22.88,31.20,42.04,48.92,52.46,38.20,21.17,21.64,22.35,22.70
     > ,21.85,22.48,21.65,24.68,32.64,34.61,25.60,28.85,29.96,31.08
     > ,29.12,27.80,29.40,28.00,25.27,24.82,23.49,25.01,21.43,20.08
     > ,25.93,24.75,23.02,21.27,20.59,19.92,24.67,27.90,24.78,22.24
     > ,22.47,23.38,22.65,20.90,19.50,21.25,22.41,23.10,21.91,21.71
     > ,28.97,31.99,36.56,42.72,31.70,26.19,26.40,26.93,27.25,26.20
     > ,25.39,25.03,26.10,26.81,28.87,29.87,29.37,28.20,33.41,38.68
     > ,46.68,36.35,30.08,30.22,30.29,30.34,30.36,31.98,30.82,29.86
     > ,29.48,30.10,27.73,25.86,23.06,22.13,19.53,19.98,17.29,16.48
     > ,17.13,17.28,16.65,14.76,14.14,15.58,16.57,16.30,15.78,15.30
     > ,12.83,11.72,12.91,14.15,12.86,11.39,12.74,14.05,14.57,16.58
     > ,13.28,11.90,12.74,11.78,11.07,9.68,12.27,13.18,14.51,13.60
     > ,12.30,11.05,10.26,10.48,10.40,10.27,10.21,10.13,10.07,9.97
     > ,10.74,11.00,10.65,10.33,9.92,9.73,9.62,9.92,12.51,10.86
     > ,9.76,9.82,10.19,10.61,11.18,11.39,12.23,11.94,11.49,10.88
     > ,9.05,10.55,13.74,15.16,14.40,13.65,12.44,11.97,12.26,12.55
     > ,12.17,11.79,11.42,10.68,10.50,12.14,11.61,10.56,11.72,8.89
     > ,7.45,8.43,10.44,9.48,6.32,12.24,18.80,16.76,14.90,13.05
     > ,11.15,7.90,8.92,9.77,11.11,13.50,16.11,18.92,21.93,14.68
     > ,11.19,8.27,8.74,9.02,9.11,11.10,10.15,7.90,7.04,5.80
     > ,6.94,8.14,9.15,8.82,9.81,10.86,11.58,13.11,16.18,16.20
     > ,16.18,16.08,11.63,7.88,9.31,10.54,10.76,10.65,12.98,11.52
     > ,10.88,10.19,9.01,8.32,7.87,7.21,7.24,7.25,6.97,8.47
     > ,9.15,10.47,12.81,16.45,14.28,10.84,10.76,9.44,10.91,11.69
     > ,9.94,10.72,11.14,12.23,13.00,13.46,13.61,12.96,11.29,9.00
     > ,6.50,9.55,11.40,12.23,13.14,15.35,17.28,17.60,17.17,16.22
     > ,15.38,14.58,13.91,11.72,10.61,10.19,9.12,8.04,7.32,6.24
     > ,7.42,10.75,14.65,16.07,17.73,19.13,21.69,24.06,25.38,27.50
     > ,20.85,13.32,8.23,9.16,9.81,11.53,12.25,11.90,10.67,9.11
     > ,12.77,15.92,18.63,20.91,22.47,23.76,20.02,16.44,13.51,10.85
     > ,11.06,11.27,11.63,11.59,11.38,10.61,8.71,6.82,5.69,6.14
     > ,7.26,9.47,9.90,9.01,9.78,10.49,7.64,8.09,8.70,6.87
     > ,5.00,4.96,4.86,4.80,4.72,4.59,7.35,8.83,8.59,8.81
     > ,7.95,3.87,8.62,10.79,10.84,10.31,7.27,6.23,4.56,4.03
     > ,4.10,3.85,3.77,3.70,5.45,6.33,5.31,4.58,4.14,5.37
     > ,9.07,10.48,8.33,6.97,6.44,5.67,4.43,3.26,3.33,3.67
     > ,4.00,4.18,4.07,3.77,4.33,4.91,5.21,3.55,2.44,2.67
     > ,2.96,3.34,3.48,3.61,3.80,3.98,3.87,3.73,3.65,3.91
     > ,4.16,4.58,5.26,5.54,5.10,4.38,3.67,2.98,3.26,3.55
     > ,3.66,3.83,4.12,4.30,3.94,3.68,3.25,2.84,2.68,3.55
     > ,4.01,4.74,4.88,5.03,5.32,5.37,5.60,5.77,5.42,5.26
     > ,5.09,4.59,4.18,3.78,3.38,3.56,3.88,4.20,4.52,4.84
     > ,5.44,5.60,5.90,5.56,5.27,4.98,4.67,4.36,4.04,3.72
     > ,3.58,4.13,5.07,5.45,6.03,7.01,8.01,8.46,8.75,8.23
     > ,7.89,7.53,7.17,6.93,6.33,5.74,5.16,4.59,4.02,3.79
     > ,3.45,3.12,3.79,5.46,7.10,8.72,10.33,11.28,10.95,10.07
     > ,9.29,8.00,6.81,5.52,4.13,4.80,6.00,7.20,8.40,7.81
     > ,6.77,5.73,4.69,6.11,7.84,7.85,7.76,6.34,4.29,3.25
     > ,3.41,3.83,4.25,4.43,4.29,3.74,3.18,2.95,3.46,4.31
     > ,5.17,5.52,6.40,7.83,7.57,7.03,5.72,7.29,8.12,8.64
     > ,8.10,6.81,5.60,4.46,3.41,2.62,4.24,5.34,7.02,8.45
     > ,9.91,11.40,11.10,10.49,10.80,9.58,8.27,7.57,6.92,6.03
     > ,5.53,4.09,3.22,3.56,4.48,5.49,5.92,4.66,2.86,5.71
     > ,9.95,14.48,16.38,15.88,14.70,13.12,11.56,7.51,5.07,2.31
     > ,3.41,4.49,5.51,6.90,7.74,8.32,5.70,10.62,13.97,17.01
     > ,20.59,15.02,10.03,5.60,4.00,2.48,2.67,2.86,2.45,2.59
     > ,2.91,2.74,16.69,17.86,15.08,9.41,15.09,22.31,18.70,16.38
     > ,15.64,12.54,9.39,6.19,4.25,2.94,6.05,10.81,15.64,20.53
     > ,23.50,24.50,30.52,35.61,38.70,36.59,31.54,26.83,22.47,18.44
     > ,14.75,12.05,8.40,6.76,6.56,5.28,2.90,8.25,22.38,35.39
     > ,43.84,38.92,30.75,22.63,14.55,10.68,6.50,1.69,3.14,6.75
     > ,13.97,21.18,28.40,35.62,42.12,27.28,22.33,29.65,24.31,19.25
     > ,17.30,14.46,11.72,9.94,5.70,1.73,4.17,6.68,9.27,11.40
     > ,7.19,5.03,7.55,20.85,35.24,42.84,38.11,33.50,29.02,24.65
     > ,20.41,12.28,8.40,4.65,1.73,7.43,11.81,15.87,19.59,24.31
     > ,14.14,7.32,2.02,10.67,18.49,18.27,17.58,16.70,15.62,13.63
     > ,11.65,9.68,7.73,5.79,3.86,1.94,8.37,14.97,21.74,28.67
     > ,35.77,40.11,36.29,26.79,17.32,7.90,4.14,5.10,6.36,5.72
     > ,4.54,2.76,4.03,2.58,1.07,1.91,2.87,7.45,9.74,12.95
     > ,19.61,18.79,20.83,18.03,12.57,7.29,2.16,1.16,1.18,1.20
     > ,1.22,2.24,3.29,4.37,5.48,4.85,5.44,4.65,2.69,1.16
     > ,1.22,1.51,1.18,1.24,1.17,1.15,1.07,1.03,0.91,0.84
     > ,0.95,1.03,0.86,1.12,1.30,0.88,1.02,0.79,1.17,0.70
     > ,0.86,1.09,0.77,0.85,1.00,0.90,0.83,1.11,0.97,1.12
     > ,1.00,0.88,1.09,1.24,0.94,0.65,0.94,1.15,1.05,0.00
     > ,0.00/

       DATA O2P_OP/0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,2.11,2.12,2.14,2.16
     > ,2.19,2.21,2.22,2.25,2.32,2.37,2.37,2.38,2.38,2.39
     > ,2.39,2.46,2.48,2.48,2.50,2.53,2.56,2.63,2.65,2.67
     > ,2.72,2.76,2.76,2.82,2.84,2.87,2.89,2.96,2.98,3.01
     > ,3.02,3.09,3.11,3.11,3.15,3.16,3.17,3.18,3.19,3.20
     > ,3.20,3.21,3.22,3.22,3.23,3.23,3.23,3.24,3.25,3.25
     > ,3.25,3.26,3.26,3.26,3.27,3.28,3.29,3.32,3.33,3.36
     > ,3.38,3.38,3.39,3.40,3.41,3.41,3.45,3.46,3.49,3.50
     > ,3.51,3.55,3.57,3.57,3.58,3.60,3.61,3.63,3.66,3.66
     > ,3.70,3.70,3.71,3.74,3.77,3.79,3.80,3.81,3.83,3.84
     > ,3.85,3.87,3.87,3.87,3.88,3.90,3.91,3.92,3.92,3.95
     > ,3.97,4.01,4.04,4.04,4.04,4.09,4.09,4.12,4.14,4.16
     > ,4.18,4.18,4.19,4.21,4.22,4.24,4.24,4.24,4.27,4.28
     > ,4.28,4.30,4.31,4.32,4.34,4.36,4.38,4.41,4.42,4.44
     > ,4.45,4.49,4.51,4.55,4.55,4.56,4.58,4.60,4.61,4.63
     > ,4.64,4.65,4.66,4.66,4.67,4.69,4.70,4.74,4.76,4.77
     > ,4.82,4.85,4.87,4.91,4.92,4.94,4.94,4.97,5.01,5.03
     > ,5.09,5.19,5.26,5.29,5.30,5.56,5.59,5.60,5.64,5.66
     > ,5.70,5.72,5.73,5.74,5.75,5.77,5.77,5.78,5.81,5.85
     > ,5.86,5.87,5.90,5.91,5.92,5.92,5.94,5.95,5.96,5.97
     > ,5.97,5.97,5.98,5.99,5.99,6.00,6.02,6.02,6.00,5.94
     > ,5.93,5.78,5.65,5.62,5.62,5.54,5.52,5.51,5.50,5.36
     > ,5.20,5.07,5.06,4.94,4.92,4.92,4.92,4.92,4.91,4.91
     > ,4.83,4.78,4.70,4.69,4.63,4.62,4.56,4.53,4.62,4.73
     > ,4.80,4.79,4.79,4.79,4.80,4.80,4.80,4.80,4.80,4.80
     > ,4.80,4.81,4.81,4.80,4.80,4.80,4.80,4.79,4.79,4.79
     > ,4.79,4.78,4.78,4.78,4.78,4.79,4.80,4.81,4.82,4.83
     > ,4.84,4.85,4.86,4.87,4.88,4.87,4.87,4.86,4.85,4.84
     > ,4.84,4.83,4.83,4.82,4.82,4.81,4.80,4.79,4.79,4.79
     > ,4.80,4.80,4.80,4.80,4.80,4.80,4.80,4.80,4.81,4.81
     > ,4.81,4.81,4.81,4.81,4.81,4.81,4.82,4.82,4.83,4.84
     > ,4.84,4.85,4.86,4.86,4.87,4.87,4.87,4.88,4.89,4.89
     > ,4.90,4.91,4.91,4.92,4.93,4.93,4.93,4.94,4.94,4.94
     > ,4.95,4.95,4.95,4.96,4.96,4.96,4.99,5.02,5.05,5.06
     > ,5.08,5.08,5.11,5.14,5.15,5.16,5.18,5.21,5.22,5.24
     > ,5.27,5.27,5.27,5.27,5.27,5.27,5.28,5.28,5.28,5.28
     > ,5.28,5.28,5.28,5.28,5.29,5.29,5.29,5.29,5.29,5.31
     > ,5.33,5.34,5.36,5.38,5.40,5.42,5.44,5.46,5.49,5.49
     > ,5.49,5.51,5.51,5.51,5.51,5.51,5.51,5.52,5.52,5.52
     > ,5.47,5.42,5.39,5.37,5.14,4.92,4.75,4.68,4.36,4.14
     > ,4.56,4.42,4.23,4.44,4.45,4.52,4.52,4.51,4.50,4.48
     > ,4.45,4.44,4.41,4.37,4.24,4.21,3.90,3.88,3.82,3.72
     > ,3.18,3.28,3.72,4.04,3.82,3.14,2.95,2.93,2.57,1.94
     > ,1.88,1.81,1.85,1.86,1.92,1.89,1.59,1.49,1.39,1.60
     > ,1.58,1.48,1.46,1.23,1.14,1.22,1.29,1.47,1.28,1.15
     > ,1.16,1.12,0.97,0.94,0.86,0.98,1.18,1.04,0.92,0.86
     > ,0.90,0.95,0.98,0.87,0.91,1.00,1.05,1.04,1.00,0.97
     > ,0.94,0.91,0.89,0.84,1.08,1.18,1.15,1.09,0.95,0.98
     > ,1.04,0.96,0.92,1.04,1.00,0.98,1.00,1.08,1.11,1.15
     > ,1.22,1.24,1.28,1.20,1.02,0.98,1.04,1.19,1.13,1.31
     > ,1.33,1.30,1.24,1.18,1.06,1.03,1.08,1.10,1.15,1.18
     > ,1.17,1.13,1.08,1.03,0.98,0.94,0.92,0.89,0.85,0.80
     > ,0.79,0.76,0.75,0.72,0.67,0.63,0.60,0.59,0.58,0.56
     > ,0.55,0.51,0.50,0.48,0.44,0.42,0.38,0.38,0.36,0.34
     > ,0.33,0.29,0.27,0.27,0.26,0.26,0.26,0.25,0.22,0.22
     > ,0.22,0.20,0.20,0.18,0.16,0.15,0.13,0.13,0.12,0.14
     > ,0.13,0.11,0.10,0.10,0.09,0.09,0.09,0.08,0.07,0.05
     > ,0.05,0.05,0.04,0.02,0.02,0.01,0.01,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/

       DATA O2P_ION/0.07,0.13,0.13,0.14,0.14,0.15,0.15,0.16,0.16,0.19
     > ,0.21,0.28,0.28,0.31,0.31,0.31,0.32,0.33,0.34,0.35
     > ,0.36,0.36,0.38,0.39,0.42,0.43,0.44,0.44,0.45,0.47
     > ,0.48,0.50,0.52,0.53,0.54,0.54,0.55,0.56,0.58,0.59
     > ,0.60,0.61,0.62,0.62,0.65,0.67,0.68,0.70,0.70,0.72
     > ,0.73,0.74,0.75,0.75,0.76,0.76,0.77,0.78,0.79,0.80
     > ,0.82,0.84,0.84,0.86,0.89,0.89,0.90,0.93,0.94,0.96
     > ,0.98,0.99,1.00,1.00,1.03,1.04,1.04,1.06,1.06,1.09
     > ,1.11,1.12,1.14,1.14,1.15,1.16,1.17,1.18,1.19,1.21
     > ,1.21,1.22,1.23,1.24,1.27,1.28,1.29,1.30,1.31,1.32
     > ,1.33,1.34,1.36,1.36,1.38,1.39,1.40,1.41,1.43,1.45
     > ,1.46,1.47,1.49,1.50,1.51,1.53,1.53,1.55,1.56,1.57
     > ,1.58,1.60,1.61,1.62,1.64,1.65,1.67,1.69,1.71,1.71
     > ,1.73,1.74,1.77,1.79,1.80,1.82,1.83,1.85,1.86,1.87
     > ,1.88,1.91,1.92,1.96,1.98,1.99,2.00,2.02,2.04,2.05
     > ,2.06,2.08,2.10,2.11,2.13,2.15,2.16,2.18,2.18,2.20
     > ,2.21,2.24,2.25,2.27,2.29,2.31,2.34,2.37,2.41,2.42
     > ,2.43,2.44,2.46,2.47,2.50,2.52,2.57,2.59,2.60,2.65
     > ,2.66,2.68,2.73,2.75,2.78,2.79,2.81,2.89,2.94,2.95
     > ,2.96,3.02,3.04,3.08,3.10,3.15,3.24,3.27,3.32,3.37
     > ,3.43,3.49,3.52,3.60,3.78,3.94,3.95,3.97,3.99,4.02
     > ,4.03,4.30,4.38,4.40,4.50,4.65,4.73,4.93,4.99,5.05
     > ,5.21,5.32,5.33,5.47,5.52,5.62,5.68,5.87,5.91,6.03
     > ,6.05,6.31,6.37,6.39,6.53,6.58,6.60,6.63,6.68,6.70
     > ,6.77,6.83,6.88,6.89,6.99,7.01,7.03,7.04,7.15,7.15
     > ,7.21,7.28,7.31,7.33,7.36,7.39,7.42,7.51,7.53,7.61
     > ,7.65,7.67,7.70,7.72,7.74,7.76,7.87,7.89,7.98,8.00
     > ,8.04,8.15,8.20,8.22,8.24,8.32,8.35,8.40,8.51,8.51
     > ,8.61,8.62,8.68,8.76,8.86,8.94,9.01,9.05,9.13,9.16
     > ,9.21,9.30,9.31,9.33,9.39,9.45,9.54,9.55,9.57,9.67
     > ,9.73,9.85,9.92,9.94,9.95,10.07,10.08,10.17,10.23,10.30
     > ,10.40,10.41,10.45,10.56,10.60,10.68,10.70,10.71,10.86,10.88
     > ,10.90,11.00,11.07,11.10,11.15,11.22,11.31,11.41,11.45,11.50
     > ,11.54,11.69,11.77,11.89,11.90,11.96,12.06,12.17,12.25,12.34
     > ,12.43,12.46,12.52,12.55,12.62,12.73,12.80,12.90,12.98,13.00
     > ,13.14,13.20,13.25,13.37,13.40,13.44,13.47,13.55,13.65,13.70
     > ,13.80,14.00,14.12,14.18,14.20,14.65,14.70,14.74,14.86,14.91
     > ,15.04,15.10,15.13,15.15,15.19,15.24,15.26,15.28,15.37,15.50
     > ,15.56,15.60,15.79,15.85,15.90,15.91,16.09,16.14,16.20,16.23
     > ,16.28,16.30,16.34,16.45,16.48,16.51,16.68,16.70,16.72,16.80
     > ,16.81,17.00,17.10,17.12,17.12,17.18,17.20,17.20,17.21,17.25
     > ,17.30,17.40,17.41,17.50,17.65,17.65,17.67,17.72,17.80,17.80
     > ,17.88,17.92,18.00,18.03,18.18,18.19,18.32,18.40,18.59,18.80
     > ,19.20,19.59,19.60,19.65,19.67,19.68,19.73,19.80,19.84,19.88
     > ,19.92,19.96,20.00,20.03,20.06,20.09,20.12,20.15,20.18,20.21
     > ,20.22,20.23,20.24,20.27,20.30,20.34,20.38,20.42,20.46,20.50
     > ,20.54,20.58,20.62,20.66,20.70,20.71,20.73,20.76,20.79,20.82
     > ,20.85,20.88,20.88,20.90,20.91,20.94,20.97,21.00,21.02,21.04
     > ,21.08,21.09,21.11,21.12,21.14,21.16,21.20,21.20,21.24,21.25
     > ,21.27,21.28,21.30,21.32,21.34,21.36,21.40,21.41,21.43,21.46
     > ,21.48,21.49,21.52,21.53,21.54,21.55,21.56,21.58,21.60,21.61
     > ,21.64,21.67,21.68,21.70,21.74,21.78,21.82,21.86,21.90,21.91
     > ,21.94,21.98,22.02,22.06,22.09,22.10,22.15,22.20,22.25,22.26
     > ,22.29,22.30,22.35,22.40,22.40,22.42,22.45,22.50,22.52,22.55
     > ,22.60,22.63,22.64,22.67,22.68,22.68,22.70,22.72,22.73,22.76
     > ,22.79,22.80,22.84,22.86,22.88,22.92,22.96,22.98,23.00,23.06
     > ,23.10,23.12,23.18,23.24,23.30,23.36,23.42,23.48,23.54,23.56
     > ,23.56,23.60,23.64,23.64,23.68,23.72,23.76,23.80,23.92,24.00
     > ,24.28,24.50,24.53,24.54,24.73,24.90,25.25,25.40,25.28,25.20
     > ,25.90,25.90,25.90,26.04,26.05,26.10,26.10,26.10,26.04,26.00
     > ,25.86,25.80,25.66,25.50,24.80,24.61,22.88,22.72,22.40,21.80
     > ,19.30,19.40,20.80,21.80,23.80,27.70,28.49,28.60,29.20,25.58
     > ,25.19,24.82,26.28,26.83,28.71,28.54,25.41,24.21,23.05,26.89
     > ,28.13,28.08,28.07,24.79,23.57,25.69,27.80,33.45,29.75,27.28
     > ,28.65,27.90,24.56,23.83,21.99,25.18,30.88,27.75,24.94,23.54
     > ,24.75,26.66,21.89,24.77,26.65,29.63,31.42,31.17,30.44,29.70
     > ,27.98,26.50,25.50,23.00,28.73,31.04,29.78,28.52,25.15,26.20
     > ,27.79,26.05,25.30,27.46,25.99,25.10,25.34,26.15,26.48,26.90
     > ,27.59,27.74,28.13,25.84,21.24,20.32,21.43,24.79,24.34,28.72
     > ,29.35,28.80,27.89,26.99,25.18,24.47,25.51,26.03,27.23,28.61
     > ,28.54,28.22,27.89,27.56,27.23,26.89,26.79,26.56,26.22,25.87
     > ,25.77,25.53,25.39,25.22,24.65,24.13,23.83,23.70,23.53,23.32
     > ,23.27,22.98,22.90,22.70,22.42,22.25,20.38,21.01,20.80,20.44
     > ,20.56,19.61,18.95,19.24,19.52,20.07,20.90,20.44,19.30,20.21
     > ,20.82,19.44,20.56,19.33,18.37,18.10,17.73,17.58,17.42,23.26
     > ,21.80,20.81,19.30,20.42,20.93,21.02,21.85,21.21,19.32,17.48
     > ,21.02,20.26,19.10,17.13,18.03,21.66,23.49,26.27,27.20,21.95
     > ,20.11,17.34,18.99,22.85,22.22,20.35,18.53,15.03,16.98,18.10
     > ,15.75,23.46,21.08,19.75,18.43,17.12,17.83,25.88,28.52,31.14
     > ,28.49,25.82,18.58,16.74,19.49,24.92,27.60,17.27,20.98,24.64
     > ,26.80,27.21,27.61,28.80,29.97,25.27,21.42,19.89,20.71,23.18
     > ,25.64,26.46,29.75,26.35,22.95,16.15,17.28,18.42,24.08,23.75
     > ,22.88,31.20,42.04,48.92,52.46,38.20,21.17,21.64,22.35,22.70
     > ,21.85,22.48,21.65,24.68,32.64,34.61,25.60,28.85,29.96,31.08
     > ,29.12,27.80,29.40,28.00,25.27,24.82,23.49,25.01,21.43,20.08
     > ,25.93,24.75,23.02,21.27,20.59,19.92,24.67,27.90,24.78,22.24
     > ,22.47,23.38,22.65,20.90,19.50,21.25,22.41,23.10,21.91,21.71
     > ,28.97,31.99,36.56,42.72,31.70,26.19,26.40,26.93,27.25,26.20
     > ,25.39,25.03,26.10,26.81,28.87,29.87,29.37,28.20,33.41,38.68
     > ,46.68,36.35,30.08,30.22,30.29,30.34,30.36,31.98,30.82,29.86
     > ,29.48,30.10,27.73,25.86,23.06,22.13,19.53,19.98,17.29,16.48
     > ,17.13,17.28,16.65,14.76,14.14,15.58,16.57,16.30,15.78,15.30
     > ,12.83,11.72,12.91,14.15,12.86,11.39,12.74,14.05,14.57,16.58
     > ,13.28,11.90,12.74,11.78,11.07,9.68,12.27,13.18,14.51,13.60
     > ,12.30,11.05,10.26,10.48,10.40,10.27,10.21,10.13,10.07,9.97
     > ,10.74,11.00,10.65,10.33,9.92,9.73,9.62,9.92,12.51,10.86
     > ,9.76,9.82,10.19,10.61,11.18,11.39,12.23,11.94,11.49,10.88
     > ,9.05,10.55,13.74,15.16,14.40,13.65,12.44,11.97,12.26,12.55
     > ,12.17,11.79,11.42,10.68,10.50,12.14,11.61,10.56,11.72,8.89
     > ,7.45,8.43,10.44,9.48,6.32,12.24,18.80,16.76,14.90,13.05
     > ,11.15,7.90,8.92,9.77,11.11,13.50,16.11,18.92,21.93,14.68
     > ,11.19,8.27,8.74,9.02,9.11,11.10,10.15,7.90,7.04,5.80
     > ,6.94,8.14,9.15,8.82,9.81,10.86,11.58,13.11,16.18,16.20
     > ,16.18,16.08,11.63,7.88,9.31,10.54,10.76,10.65,12.98,11.52
     > ,10.88,10.19,9.01,8.32,7.87,7.21,7.24,7.25,6.97,8.47
     > ,9.15,10.47,12.81,16.45,14.28,10.84,10.76,9.44,10.91,11.69
     > ,9.94,10.72,11.14,12.23,13.00,13.46,13.61,12.96,11.29,9.00
     > ,6.50,9.55,11.40,12.23,13.14,15.35,17.28,17.60,17.17,16.22
     > ,15.38,14.58,13.91,11.72,10.61,10.19,9.12,8.04,7.32,6.24
     > ,7.42,10.75,14.65,16.07,17.73,19.13,21.69,24.06,25.38,27.50
     > ,20.85,13.32,8.23,9.16,9.81,11.53,12.25,11.90,10.67,9.11
     > ,12.77,15.92,18.63,20.91,22.47,23.76,20.02,16.44,13.51,10.85
     > ,11.06,11.27,11.63,11.59,11.38,10.61,8.71,6.82,5.69,6.14
     > ,7.26,9.47,9.90,9.01,9.78,10.49,7.64,8.09,8.70,6.87
     > ,5.00,4.96,4.86,4.80,4.72,4.59,7.35,8.83,8.59,8.81
     > ,7.95,3.87,8.62,10.79,10.84,10.31,7.27,6.23,4.56,4.03
     > ,4.10,3.85,3.77,3.70,5.45,6.33,5.31,4.58,4.14,5.37
     > ,9.07,10.48,8.33,6.97,6.44,5.67,4.43,3.26,3.33,3.67
     > ,4.00,4.18,4.07,3.77,4.33,4.91,5.21,3.55,2.44,2.67
     > ,2.96,3.34,3.48,3.61,3.80,3.98,3.87,3.73,3.65,3.91
     > ,4.16,4.58,5.26,5.54,5.10,4.38,3.67,2.98,3.26,3.55
     > ,3.66,3.83,4.12,4.30,3.94,3.68,3.25,2.84,2.68,3.55
     > ,4.01,4.74,4.88,5.03,5.32,5.37,5.60,5.77,5.42,5.26
     > ,5.09,4.59,4.18,3.78,3.38,3.56,3.88,4.20,4.52,4.84
     > ,5.44,5.60,5.90,5.56,5.27,4.98,4.67,4.36,4.04,3.72
     > ,3.58,4.13,5.07,5.45,6.03,7.01,8.01,8.46,8.75,8.23
     > ,7.89,7.53,7.17,6.93,6.33,5.74,5.16,4.59,4.02,3.79
     > ,3.45,3.12,3.79,5.46,7.10,8.72,10.33,11.28,10.59,10.07
     > ,9.29,8.00,6.81,5.52,4.13,4.80,6.00,7.20,8.40,7.81
     > ,6.77,5.73,4.69,6.11,7.84,7.85,7.76,6.34,4.29,3.25
     > ,3.41,3.83,4.25,4.43,4.29,3.74,3.18,2.95,3.46,4.31
     > ,5.17,5.52,6.40,7.83,7.57,7.03,5.72,7.29,8.12,8.64
     > ,8.10,6.81,5.60,4.46,3.41,2.62,4.24,5.34,7.02,8.45
     > ,9.91,11.40,11.10,10.49,10.80,9.58,8.27,7.57,6.92,6.03
     > ,5.53,4.09,3.22,3.56,4.48,5.49,5.92,4.66,2.86,5.71
     > ,9.95,14.48,16.38,15.88,14.70,13.12,11.56,7.51,5.07,2.31
     > ,3.41,4.49,5.51,6.90,7.74,8.32,5.70,10.62,13.97,17.01
     > ,20.59,15.02,10.03,5.60,4.00,2.48,2.67,2.86,2.45,2.59
     > ,2.91,2.74,16.69,17.86,15.08,9.41,15.09,22.31,18.70,16.38
     > ,15.64,12.54,9.39,6.19,4.25,2.94,6.05,10.81,15.64,20.53
     > ,23.50,24.50,30.52,35.61,38.70,36.59,31.54,26.83,22.47,18.44
     > ,14.75,12.05,8.40,6.76,6.56,5.28,2.90,8.25,22.38,35.39
     > ,43.84,38.92,30.75,22.63,14.55,10.68,6.50,1.69,3.14,6.75
     > ,13.97,21.18,28.40,35.62,42.12,27.28,22.33,29.65,24.31,19.25
     > ,17.30,14.46,11.72,9.94,5.70,1.73,4.17,6.68,9.27,11.40
     > ,7.19,5.03,7.55,20.85,35.24,42.84,38.11,33.50,29.02,24.65
     > ,20.41,12.28,8.40,4.65,1.73,7.43,11.81,15.87,19.59,24.31
     > ,14.14,7.32,2.02,10.67,18.49,18.27,17.58,16.70,15.62,13.63
     > ,11.65,9.68,7.73,5.79,3.86,1.94,8.37,14.97,21.74,28.67
     > ,35.77,40.11,36.29,26.79,17.32,7.90,4.14,5.10,6.36,5.72
     > ,4.54,2.76,4.03,2.58,1.07,1.91,2.87,7.45,9.74,12.95
     > ,19.61,18.79,20.83,18.03,12.57,7.29,2.16,1.16,1.18,1.20
     > ,1.22,2.24,3.29,4.37,5.48,4.85,5.44,4.65,2.69,1.16
     > ,1.22,1.51,1.18,1.24,1.17,1.15,1.07,1.03,0.91,0.84
     > ,0.95,1.03,0.86,1.12,1.30,0.88,1.02,0.79,1.17,0.70
     > ,0.86,1.09,0.77,0.85,1.00,0.90,0.83,1.11,0.97,1.12
     > ,1.00,0.88,1.09,1.24,0.94,0.65,0.94,1.15,1.05,0.00
     > ,0.00/
                   
       DATA NP/0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.04,0.04,0.05
     > ,0.05,0.07,0.07,0.08,0.08,0.09,0.09,0.09,0.09,0.09
     > ,0.10,0.10,0.10,0.10,0.11,0.11,0.11,0.12,0.12,0.12
     > ,0.13,0.13,0.14,0.14,0.14,0.14,0.14,0.14,0.15,0.15
     > ,0.15,0.16,0.16,0.16,0.17,0.17,0.18,0.18,0.18,0.18
     > ,0.19,0.19,0.19,0.19,0.20,0.20,0.20,0.20,0.20,0.21
     > ,0.21,0.21,0.22,0.22,0.22,0.23,0.23,0.23,0.24,0.24
     > ,0.25,0.25,0.25,0.26,0.26,0.27,0.27,0.27,0.27,0.28
     > ,0.28,0.29,0.29,0.29,0.29,0.29,0.30,0.30,0.30,0.31
     > ,0.31,0.31,0.31,0.32,0.32,0.32,0.33,0.33,0.33,0.34
     > ,0.34,0.34,0.35,0.35,0.35,0.36,0.36,0.36,0.36,0.37
     > ,0.37,0.37,0.38,0.38,0.38,0.38,0.39,0.39,0.39,0.39
     > ,0.40,0.40,0.40,0.41,0.41,0.41,0.41,0.42,0.42,0.43
     > ,0.43,0.43,0.44,0.44,0.45,0.45,0.45,0.46,0.46,0.46
     > ,0.47,0.47,0.47,0.48,0.49,0.49,0.49,0.50,0.50,0.51
     > ,0.51,0.51,0.52,0.52,0.53,0.53,0.54,0.54,0.54,0.54
     > ,0.55,0.56,0.56,0.56,0.57,0.58,0.58,0.59,0.60,0.60
     > ,0.61,0.61,0.61,0.62,0.63,0.63,0.65,0.65,0.66,0.67
     > ,0.67,0.68,0.69,0.70,0.71,0.71,0.72,0.74,0.76,0.76
     > ,0.76,0.78,0.79,0.80,0.81,0.83,0.86,0.86,0.88,0.89
     > ,0.90,0.92,0.93,0.94,0.99,1.03,1.04,1.04,1.05,1.05
     > ,1.06,1.13,1.15,1.15,1.17,1.22,1.24,1.30,1.32,1.34
     > ,1.39,1.43,1.43,1.48,1.49,1.52,1.54,1.65,1.68,1.74
     > ,1.76,1.91,1.95,1.96,2.03,2.05,2.07,2.08,2.10,2.11
     > ,2.15,2.18,2.20,2.21,2.25,2.27,2.28,2.28,2.34,2.34
     > ,2.37,2.41,2.43,2.43,2.45,2.46,2.47,2.51,2.52,2.55
     > ,2.57,2.57,2.59,2.60,2.61,2.61,2.66,2.67,2.71,2.72
     > ,2.73,2.78,2.80,2.81,2.82,2.86,2.87,2.89,2.94,2.95
     > ,2.99,3.00,3.02,3.06,3.11,3.14,3.18,3.19,3.23,3.25
     > ,3.27,3.31,3.32,3.33,3.35,3.38,3.42,3.43,3.43,3.47
     > ,3.50,3.55,3.58,3.59,3.60,3.65,3.65,3.69,3.72,3.75
     > ,3.79,3.80,3.81,3.84,3.86,3.88,3.89,3.89,3.94,3.94
     > ,3.95,3.98,4.00,4.01,4.04,4.07,4.10,4.14,4.16,4.18
     > ,4.20,4.26,4.30,4.34,4.35,4.37,4.40,4.44,4.47,4.50
     > ,4.53,4.54,4.56,4.57,4.60,4.64,4.66,4.70,4.73,4.74
     > ,4.79,4.81,4.83,4.87,4.88,4.90,4.91,4.94,4.98,4.99
     > ,5.03,5.09,5.13,5.15,5.15,5.30,5.31,5.33,5.37,5.39
     > ,5.44,5.47,5.48,5.49,5.50,5.52,5.53,5.54,5.57,5.62
     > ,5.63,5.64,5.72,5.75,5.77,5.78,5.85,5.88,5.90,5.93
     > ,5.97,5.98,6.01,6.10,6.12,6.15,6.28,6.30,6.32,6.40
     > ,6.41,6.60,6.75,6.78,6.79,6.87,6.89,6.90,6.92,7.05
     > ,7.20,7.35,7.36,7.50,7.65,7.65,7.67,7.72,7.80,7.80
     > ,7.92,7.98,8.10,8.12,8.21,8.22,8.30,8.35,8.47,8.60
     > ,8.85,9.10,9.10,9.13,9.14,9.15,9.18,9.23,9.25,9.28
     > ,9.30,9.33,9.35,9.38,9.40,9.43,9.45,9.48,9.50,9.53
     > ,9.53,9.54,9.55,9.58,9.60,9.62,9.64,9.66,9.68,9.70
     > ,9.72,9.74,9.76,9.78,9.80,9.81,9.82,9.84,9.86,9.88
     > ,9.90,9.92,9.92,9.93,9.94,9.96,9.98,10.00,10.01,10.02
     > ,10.05,10.06,10.07,10.07,10.09,10.10,10.13,10.13,10.15,10.16
     > ,10.17,10.18,10.19,10.20,10.22,10.23,10.25,10.26,10.27,10.30
     > ,10.32,10.32,10.35,10.36,10.37,10.38,10.39,10.40,10.41,10.43
     > ,10.45,10.48,10.48,10.50,10.52,10.53,10.55,10.56,10.57,10.58
     > ,10.59,10.61,10.62,10.64,10.65,10.65,10.66,10.68,10.69,10.70
     > ,10.71,10.71,10.73,10.74,10.74,10.75,10.76,10.77,10.78,10.78
     > ,10.80,10.82,10.82,10.83,10.84,10.84,10.85,10.86,10.87,10.88
     > ,10.89,10.90,10.92,10.93,10.94,10.96,10.98,10.99,11.00,11.02
     > ,11.03,11.04,11.06,11.08,11.10,11.12,11.14,11.16,11.18,11.19
     > ,11.19,11.20,11.22,11.22,11.23,11.24,11.26,11.27,11.32,11.35
     > ,11.43,11.50,11.51,11.51,11.56,11.60,11.67,11.70,11.71,11.72
     > ,11.74,11.75,11.76,11.77,11.77,11.78,11.79,11.80,11.80,11.80
     > ,11.80,11.80,11.80,11.80,11.80,11.80,11.78,11.78,11.78,11.77
     > ,11.76,11.75,11.74,11.73,11.73,11.72,11.70,11.70,11.74,11.78
     > ,11.78,11.79,11.80,11.80,11.77,11.76,11.72,11.70,11.69,11.68
     > ,11.64,11.60,11.59,11.57,11.57,11.56,11.55,11.53,11.52,11.52
     > ,11.51,11.50,13.90,12.10,11.25,10.40,10.67,10.43,12.92,12.71
     > ,12.19,11.92,11.78,11.70,11.50,11.33,11.23,11.22,11.17,11.12
     > ,11.00,14.60,14.31,13.57,12.84,12.54,12.32,12.10,12.04,12.02
     > ,12.00,11.94,11.92,11.86,11.82,11.80,11.78,11.70,11.60,11.47
     > ,11.26,11.22,11.09,10.97,10.73,10.68,10.63,10.49,10.27,10.10
     > ,10.68,12.41,15.30,14.66,13.38,12.87,12.10,12.10,12.10,12.10
     > ,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10
     > ,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10
     > ,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10
     > ,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10
     > ,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10
     > ,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10
     > ,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,12.10,11.96
     > ,11.90,11.82,11.74,11.65,11.62,11.54,11.46,11.29,11.15,11.06
     > ,10.98,10.70,10.28,10.14,10.00,11.28,12.56,16.41,17.69,18.98
     > ,20.90,20.20,18.32,17.85,17.14,15.73,15.42,15.10,14.90,14.70
     > ,14.58,14.50,14.42,14.30,14.18,13.94,13.74,13.66,13.62,13.50
     > ,13.48,13.47,13.44,13.43,13.42,13.39,13.37,13.36,13.29,13.26
     > ,13.25,13.22,13.19,13.15,13.13,13.09,13.04,13.01,12.97,12.95
     > ,12.88,12.86,12.82,12.80,12.74,12.73,12.67,12.63,12.61,12.60
     > ,12.56,12.53,12.47,12.45,12.41,12.40,12.38,12.35,12.31,12.30
     > ,12.27,12.24,12.20,12.17,12.16,12.14,12.10,12.09,12.08,12.07
     > ,12.06,12.06,12.05,12.04,12.03,12.01,12.00,11.99,11.98,11.98
     > ,11.96,11.96,11.95,11.94,11.93,11.92,11.91,11.90,11.89,11.88
     > ,11.88,11.86,11.86,11.85,11.84,11.83,11.81,11.81,11.80,11.80
     > ,11.79,11.78,11.77,11.75,11.74,11.73,11.72,11.70,11.69,11.68
     > ,11.67,11.66,11.65,11.64,11.63,11.62,11.60,11.60,11.59,11.59
     > ,11.58,11.58,11.58,11.57,11.57,11.57,11.56,11.56,11.56,11.55
     > ,11.54,11.54,11.53,11.53,11.53,11.52,11.52,11.51,11.51,11.50
     > ,11.49,11.49,11.48,11.48,11.48,11.47,11.46,11.46,11.45,11.45
     > ,11.44,11.44,11.44,11.44,11.43,11.43,11.43,11.43,11.42,11.42
     > ,11.42,11.42,11.41,11.41,11.41,11.40,11.40,11.39,11.39,11.38
     > ,11.38,11.37,11.37,11.36,11.35,11.35,11.35,11.34,11.34,11.33
     > ,11.33,11.32,11.32,11.32,11.31,11.31,11.30,11.30,11.30,11.29
     > ,11.29,11.28,11.28,11.28,11.27,11.27,11.27,11.26,11.26,11.25
     > ,11.25,11.24,11.24,11.23,11.23,11.22,11.21,11.20,11.20,11.19
     > ,11.18,11.18,11.17,11.16,11.16,11.15,11.15,11.14,11.13,11.12
     > ,11.12,11.12,11.11,11.10,11.10,11.10,11.09,11.09,11.08,11.08
     > ,11.07,11.06,11.06,11.05,11.05,11.04,11.04,11.03,11.03,11.03
     > ,11.02,11.01,11.01,11.00,10.99,10.99,10.98,10.98,10.97,10.97
     > ,10.96,10.95,10.94,10.94,10.94,10.93,10.93,10.92,10.91,10.91
     > ,10.91,10.90,10.90,10.89,10.88,10.88,10.87,10.86,10.85,10.83
     > ,10.83,10.82,10.81,10.80,10.80,10.79,10.78,10.78,10.77,10.77
     > ,10.77,10.76,10.76,10.76,10.76,10.75,10.75,10.75,10.75,10.74
     > ,10.74,10.74,10.74,10.73,10.72,10.72,10.72,10.71,10.71,10.71
     > ,10.70,10.70,10.70,10.69,10.69,10.69,10.69,10.69,10.68,10.68
     > ,10.68,10.68,10.68,10.67,10.67,10.66,10.66,10.66,10.66,10.65
     > ,10.65,10.64,10.64,10.63,10.63,10.63,10.63,10.62,10.62,10.62
     > ,10.62,10.62,10.61,10.61,10.60,10.60,10.59,10.58,10.57,10.57
     > ,10.56,10.54,10.54,10.52,10.51,10.50,10.49,10.49,10.48,10.47
     > ,10.46,10.46,10.45,10.44,10.44,10.43,10.42,10.41,10.41,10.40
     > ,10.40,10.38,10.36,10.35,10.34,10.34,10.33,10.33,10.32,10.32
     > ,10.31,10.30,10.30,10.29,10.28,10.27,10.26,10.25,10.24,10.24
     > ,10.23,10.22,10.20,10.16,10.15,10.13,10.10,10.06,10.02,9.99
     > ,9.95,9.93,9.92,9.89,9.85,9.81,9.79,9.71,9.67,9.64
     > ,9.60,9.57,9.56,9.53,9.50,9.47,9.45,9.43,9.42,9.41
     > ,9.40,5.96,0.40,0.39,0.37,0.34,0.31,0.29,0.26,0.23
     > ,0.22,0.20,0.17,0.15,0.13,0.11,0.09,0.06,0.05,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
     > ,0.00/
	 RETURN
	 END